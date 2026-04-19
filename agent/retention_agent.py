"""
Retention Intelligence Agent — Agentic AI Core.

Architecture:
    User Input
        → _extract_risk_factors()      [deterministic feature analysis]
        → RAGRetriever.retrieve()      [FAISS k-NN: similar historical cases]
        → LLM (Groq/Llama-3) via LangChain   [reasoning + recommendations]
        → _parse_llm_response()        [structured output extraction]
        → Fallback: rule-based         [if GROQ_API_KEY not set]

LLM: llama-3.1-8b-instant via Groq (free tier)
RAG: FAISS + sentence-transformers/all-MiniLM-L6-v2
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from dotenv import load_dotenv

from rag.retriever import RAGRetriever
from agent.prompts import RETENTION_ANALYSIS_TEMPLATE, SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)


class RetentionAgent:
    """
    LangChain-powered Agentic Retention Analyst.

    Workflow:
        1. Receives churn prediction + probability from ML model
        2. Extracts deterministic risk factors from customer features
        3. Retrieves k nearest historical cases via FAISS RAG
        4. Calls Groq LLM (Llama-3.1) with RAG-augmented prompt
        5. Parses + structures LLM response
        6. Falls back to rule-based logic if no API key present
    """

    def __init__(self, rag_retriever: RAGRetriever, feature_names: List[str]):
        self.rag_retriever = rag_retriever
        self.feature_names = feature_names
        self.llm = self._initialize_llm()

    # ─── LLM Initialisation ────────────────────────────────────────────────────

    def _initialize_llm(self):
        """
        Initialise Groq LLM via LangChain.
        Returns None (fallback mode) if GROQ_API_KEY is not set.
        """
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_api_key:
            logger.warning(
                "GROQ_API_KEY not set — RetentionAgent running in rule-based fallback mode. "
                "Set GROQ_API_KEY in .env to enable LLM-powered analysis."
            )
            return None

        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.3,      # Low temp → consistent, professional output
                max_tokens=1024,
                timeout=30,
            )
            logger.info("✅ Groq LLM initialised (llama-3.1-8b-instant)")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialise Groq LLM: {e}")
            return None

    # ─── Main Entry Point ──────────────────────────────────────────────────────

    def analyze_churn_risk(
        self,
        customer_data: Dict[str, Any],
        churn_probability: float,
    ) -> Dict[str, Any]:
        """
        Full agentic analysis pipeline.

        Args:
            customer_data:      Customer feature dictionary
            churn_probability:  ML model churn probability (0-1)

        Returns:
            Analysis dict with risk_factors, recommendations,
            retention_budget, llm_reasoning, llm_used flag
        """
        logger.info(f"Agentic analysis — churn_prob={churn_probability:.2%}")

        risk_category = self._categorize_risk(churn_probability)

        # Step 1 — Deterministic risk factor extraction
        risk_factors = self._extract_risk_factors(customer_data, churn_probability)

        # Step 2 — RAG: retrieve similar historical cases
        similar_cases = self.rag_retriever.retrieve_similar_cases(customer_data, k=5)

        # Step 3 — LLM reasoning (or rule-based fallback)
        if self.llm is not None:
            recommendations, llm_reasoning = self._llm_analysis(
                customer_data, churn_probability, risk_category,
                risk_factors, similar_cases
            )
            llm_used = True
        else:
            recommendations = self._rule_based_recommendations(
                customer_data, risk_category, risk_factors, similar_cases
            )
            llm_reasoning = None
            llm_used = False

        # Step 4 — Budget calculation
        retention_budget = self._calculate_retention_budget(
            churn_probability, customer_data
        )

        return {
            "risk_category":    risk_category,
            "churn_probability": float(churn_probability),
            "risk_factors":     risk_factors,
            "similar_cases":    similar_cases,
            "recommendations":  recommendations,
            "retention_budget": retention_budget,
            "llm_reasoning":    llm_reasoning,
            "llm_used":         llm_used,
        }

    # ─── LLM Analysis ──────────────────────────────────────────────────────────

    def _llm_analysis(
        self,
        customer_data: Dict[str, Any],
        churn_probability: float,
        risk_category: str,
        risk_factors: List[Dict[str, str]],
        similar_cases: List[Tuple[Dict, float]],
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        Call Groq LLM with RAG-augmented prompt.
        Returns (structured_recommendations, raw_llm_reasoning).
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        # Format customer profile for prompt
        customer_profile = self._format_customer_profile(customer_data)

        # Format risk factors for prompt
        risk_factors_text = "\n".join(
            f"  • {f['factor']} [{f['impact']}]: {f['reason']}"
            for f in risk_factors
        ) or "  • No high-impact risk factors automatically detected."

        # Format RAG similar cases for prompt
        similar_cases_text = self._format_similar_cases(similar_cases)

        # Build prompt using template
        human_prompt = RETENTION_ANALYSIS_TEMPLATE.format(
            churn_probability=f"{churn_probability * 100:.1f}",
            risk_level=risk_category,
            customer_profile=customer_profile,
            risk_factors=risk_factors_text,
            similar_cases=similar_cases_text,
            monthly_charges=customer_data.get("MonthlyCharges", 50),
            tenure=customer_data.get("tenure", 0),
        )

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ]
            response = self.llm.invoke(messages)
            raw_text = response.content
            logger.info("✅ Groq LLM response received")

            # Parse structured recommendations from LLM output
            recommendations = self._parse_llm_recommendations(raw_text)
            return recommendations, raw_text

        except Exception as e:
            logger.error(f"LLM call failed: {e} — falling back to rule-based")
            fallback = self._rule_based_recommendations(
                customer_data, risk_category, risk_factors, similar_cases
            )
            return fallback, f"[LLM unavailable: {e}]"

    def _parse_llm_recommendations(self, llm_text: str) -> List[Dict[str, str]]:
        """
        Parse LLM output into structured recommendation dicts.
        Gracefully handles any LLM formatting variation.
        """
        recommendations = []
        lines = llm_text.split("\n")

        for line in lines:
            line = line.strip()
            # Match lines that look like "Action N [PRIORITY: X]: ..."
            if line.startswith("- Action") and "[PRIORITY:" in line.upper():
                try:
                    # Extract priority
                    priority = "MEDIUM"
                    if "HIGH" in line.upper():
                        priority = "HIGH"
                    elif "LOW" in line.upper():
                        priority = "LOW"

                    # Split action from impact
                    parts = line.split("|")
                    action_part = parts[0].strip(" -")
                    # Clean up "Action N [PRIORITY: X]: " prefix
                    colon_pos = action_part.find("]: ")
                    if colon_pos != -1:
                        action_part = action_part[colon_pos + 3:]

                    impact_part = parts[1].strip() if len(parts) > 1 else "Reduces churn risk"
                    if impact_part.lower().startswith("expected impact:"):
                        impact_part = impact_part[16:].strip()

                    recommendations.append({
                        "priority":        priority,
                        "action":          action_part[:80],
                        "description":     action_part,
                        "expected_impact": impact_part,
                    })
                except Exception:
                    continue

        # If parsing failed, return a single catch-all from LLM text
        if not recommendations:
            # Extract up to 3 meaningful lines as generic recommendations
            content_lines = [
                l.strip(" -•*") for l in lines
                if len(l.strip()) > 30 and not l.strip().startswith("#")
            ]
            for i, cl in enumerate(content_lines[:3]):
                recommendations.append({
                    "priority":        "MEDIUM",
                    "action":          f"AI Recommendation {i + 1}",
                    "description":     cl,
                    "expected_impact": "Reduces churn probability",
                })

        return recommendations[:5]

    # ─── Helper Formatters ─────────────────────────────────────────────────────

    def _format_customer_profile(self, customer_data: Dict[str, Any]) -> str:
        """Format customer data as a readable summary for the LLM prompt."""
        lines = [
            f"  Contract Type:    {customer_data.get('Contract', 'N/A')}",
            f"  Tenure:           {customer_data.get('tenure', 0)} months",
            f"  Monthly Charges:  ${customer_data.get('MonthlyCharges', 0):.2f}",
            f"  Total Charges:    ${customer_data.get('TotalCharges', 0):.2f}",
            f"  Internet Service: {customer_data.get('InternetService', 'N/A')}",
            f"  Tech Support:     {customer_data.get('TechSupport', 'N/A')}",
            f"  Payment Method:   {customer_data.get('PaymentMethod', 'N/A')}",
            f"  Paperless Billing:{customer_data.get('PaperlessBilling', 'N/A')}",
            f"  Senior Citizen:   {'Yes' if customer_data.get('SeniorCitizen') == 1 else 'No'}",
            f"  Partner:          {customer_data.get('Partner', 'N/A')}",
            f"  Dependents:       {customer_data.get('Dependents', 'N/A')}",
        ]
        return "\n".join(lines)

    def _format_similar_cases(
        self, similar_cases: List[Tuple[Dict, float]]
    ) -> str:
        """Format RAG-retrieved cases for the LLM prompt."""
        if not similar_cases:
            return "  No similar cases retrieved from knowledge base."

        lines = []
        for i, (case, similarity) in enumerate(similar_cases[:3], 1):
            features = case.get("customer_features", {})
            outcome = "CHURNED" if case.get("churn_outcome") else "RETAINED"
            prob = case.get("churn_probability", 0)
            strategies = case.get("retention_strategies", [])
            strategy_text = strategies[0] if strategies else "N/A"

            lines.append(
                f"  Case {i} (similarity: {similarity:.0%}) → {outcome} "
                f"[prob: {prob:.1%}]\n"
                f"    Profile: tenure={features.get('tenure','?')}mo, "
                f"contract={features.get('Contract','?')}, "
                f"charges=${features.get('MonthlyCharges', 0):.0f}/mo\n"
                f"    Strategy used: {strategy_text}"
            )
        return "\n".join(lines)

    # ─── Risk Categorisation ───────────────────────────────────────────────────

    def _categorize_risk(self, probability: float) -> str:
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        return "HIGH"

    # ─── Deterministic Risk Factors ────────────────────────────────────────────

    def _extract_risk_factors(
        self,
        customer_data: Dict[str, Any],
        churn_probability: float,
    ) -> List[Dict[str, str]]:
        """Rule-based risk factor extraction (always runs, feeds into LLM prompt)."""
        factors = []

        if customer_data.get("Contract") == "Month-to-month":
            factors.append({
                "factor": "Month-to-month contract",
                "impact": "HIGH",
                "reason": "No long-term commitment — easiest to cancel",
            })

        monthly = customer_data.get("MonthlyCharges", 0)
        if monthly > 80:
            factors.append({
                "factor": "High monthly charges",
                "impact": "MEDIUM",
                "reason": f"${monthly:.2f}/mo may strain budget vs. competitors",
            })

        if customer_data.get("TechSupport") == "No":
            factors.append({
                "factor": "No technical support subscription",
                "impact": "MEDIUM",
                "reason": "Unresolved issues increase frustration and churn likelihood",
            })

        tenure = customer_data.get("tenure", 0)
        if tenure < 6:
            factors.append({
                "factor": "New customer (low tenure)",
                "impact": "HIGH",
                "reason": f"Only {tenure} month(s) — early-stage customers churn at 2× rate",
            })

        if customer_data.get("InternetService") == "Fiber optic":
            factors.append({
                "factor": "Fiber optic service",
                "impact": "MEDIUM",
                "reason": "Fiber customers show higher churn — often linked to pricing dissatisfaction",
            })

        if customer_data.get("PaymentMethod") == "Electronic check":
            factors.append({
                "factor": "Electronic check payment",
                "impact": "LOW",
                "reason": "Correlated with higher churn — less \"sticky\" than auto-pay",
            })

        if customer_data.get("OnlineSecurity") == "No":
            factors.append({
                "factor": "No online security service",
                "impact": "LOW",
                "reason": "Missing value-added service reduces switching cost",
            })

        return factors[:4]

    # ─── Rule-Based Fallback ───────────────────────────────────────────────────

    def _rule_based_recommendations(
        self,
        customer_data: Dict[str, Any],
        risk_category: str,
        risk_factors: List[Dict[str, str]],
        similar_cases: List[Tuple[Dict, float]],
    ) -> List[Dict[str, str]]:
        """Deterministic recommendations used when LLM is unavailable."""
        recs = []

        if customer_data.get("Contract") == "Month-to-month":
            recs.append({
                "priority": "HIGH",
                "action": "Contract upgrade incentive",
                "description": "Offer 15–20% discount for committing to a 1-year contract",
                "expected_impact": "Reduces churn probability by ~30%",
            })

        if customer_data.get("TechSupport") == "No" and risk_category in ("MEDIUM", "HIGH"):
            recs.append({
                "priority": "HIGH",
                "action": "3-month free tech support trial",
                "description": "Proactively enrol customer in tech support — no charge for 90 days",
                "expected_impact": "Reduces frustration-driven churn",
            })

        if customer_data.get("MonthlyCharges", 0) > 80 and risk_category == "HIGH":
            recs.append({
                "priority": "MEDIUM",
                "action": "Bill optimisation review call",
                "description": "Schedule CSR call to review and restructure service bundle",
                "expected_impact": "Potential 10–15% bill reduction",
            })

        if customer_data.get("tenure", 0) < 6:
            recs.append({
                "priority": "HIGH",
                "action": "Onboarding success check-in",
                "description": "Dedicated welcome call + personalised onboarding guide",
                "expected_impact": "Reduces early-tenure churn by up to 25%",
            })

        if risk_category == "HIGH":
            recs.append({
                "priority": "MEDIUM",
                "action": "Loyalty rewards enrolment",
                "description": "Enrol in loyalty programme with points and exclusive offers",
                "expected_impact": "Increases emotional brand attachment",
            })

        if similar_cases:
            for case, _ in similar_cases[:2]:
                if not case.get("churn_outcome"):
                    strategies = case.get("retention_strategies", [])
                    if strategies:
                        recs.append({
                            "priority": "MEDIUM",
                            "action": "RAG-informed strategy",
                            "description": strategies[0],
                            "expected_impact": "Based on similar retained customer outcome",
                        })
                        break

        return recs[:5]

    # ─── Budget Calculation ────────────────────────────────────────────────────

    def _calculate_retention_budget(
        self,
        churn_probability: float,
        customer_data: Dict[str, Any],
    ) -> float:
        try:
            monthly = float(customer_data.get("MonthlyCharges", 50))
        except (ValueError, TypeError):
            monthly = 50.0

        # Budget = 15% of monthly × risk multiplier
        # High risk → up to 45% of monthly bill committed to retention
        risk_multiplier = (churn_probability * 2) + 0.5
        return round(monthly * 0.15 * risk_multiplier, 2)

    # ─── Report Generator ──────────────────────────────────────────────────────

    def generate_summary_report(self, analysis: Dict[str, Any]) -> str:
        """Generate formatted text report from analysis dict."""
        prob     = analysis["churn_probability"]
        risk_cat = analysis["risk_category"]
        llm_used = analysis.get("llm_used", False)

        lines = [
            "═" * 60,
            "  🛡️  CHURNPREDICTOR AI — RETENTION ANALYSIS REPORT",
            "═" * 60,
            f"  Risk Level: {risk_cat}  |  Churn Probability: {prob:.1%}",
            f"  Analysis Engine: {'🤖 Groq LLM (Llama-3.1)' if llm_used else '📐 Rule-Based Fallback'}",
            "─" * 60,
        ]

        if analysis.get("risk_factors"):
            lines.append("\n⚠️  KEY RISK FACTORS:")
            for f in analysis["risk_factors"]:
                lines.append(f"  [{f['impact']}] {f['factor']}")
                lines.append(f"    → {f['reason']}")

        if analysis.get("recommendations"):
            lines.append("\n✅ RECOMMENDED ACTIONS:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                lines.append(f"\n  {i}. [{rec['priority']}] {rec['action']}")
                lines.append(f"     {rec['description']}")
                lines.append(f"     📈 Impact: {rec['expected_impact']}")

        lines.append(f"\n💰 RETENTION BUDGET: ${analysis['retention_budget']:.2f}")

        if llm_used and analysis.get("llm_reasoning"):
            lines.append("\n─" * 60)
            lines.append("🤖 LLM REASONING (Groq / Llama-3.1-8b-instant):")
            lines.append("─" * 60)
            lines.append(analysis["llm_reasoning"])

        lines.append("\n" + "═" * 60)
        return "\n".join(lines)
