"""
Prompt templates for ChurnPredictor AI RetentionAgent.
Uses LangChain PromptTemplate for structured LLM interactions.
"""

from langchain_core.prompts import PromptTemplate

# ─── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Customer Retention Analyst AI for a telecom company.
Your role is to analyse customer churn risk and generate actionable, personalised retention strategies.
Always be specific, data-driven, and business-focused. Format responses clearly."""


# ─── Retention Analysis Prompt ─────────────────────────────────────────────────
RETENTION_ANALYSIS_TEMPLATE = PromptTemplate(
    input_variables=[
        "churn_probability", "risk_level", "customer_profile",
        "risk_factors", "similar_cases", "monthly_charges", "tenure"
    ],
    template="""You are an expert Customer Retention Analyst AI.

## Customer Churn Analysis Request

**Churn Probability:** {churn_probability}% ({risk_level} RISK)

**Customer Profile:**
{customer_profile}

**Identified Risk Factors:**
{risk_factors}

**Similar Historical Cases (from RAG knowledge base):**
{similar_cases}

## Your Task
Based on the churn probability, customer profile, risk factors, and similar historical cases above:

1. **Root Cause Analysis** (2-3 sentences): What are the primary drivers of churn risk for this customer?

2. **Priority Retention Actions** (exactly 3 specific actions):
   - Action 1 [PRIORITY: HIGH/MEDIUM/LOW]: <specific action> | Expected impact: <measurable outcome>
   - Action 2 [PRIORITY: HIGH/MEDIUM/LOW]: <specific action> | Expected impact: <measurable outcome>  
   - Action 3 [PRIORITY: HIGH/MEDIUM/LOW]: <specific action> | Expected impact: <measurable outcome>

3. **Retention Budget Justification** (1 sentence): Justify the recommended retention spend relative to the ${monthly_charges}/month revenue and {tenure}-month tenure.

4. **Predicted Outcome** (1 sentence): If these strategies are executed within 30 days, what is the expected churn risk reduction?

Be specific, quantitative where possible, and actionable. Do not be generic."""
)


# ─── Risk Factor Summary Prompt ────────────────────────────────────────────────
RISK_SUMMARY_TEMPLATE = PromptTemplate(
    input_variables=["risk_factors", "churn_probability"],
    template="""Given these churn risk factors for a customer with {churn_probability}% churn probability:

{risk_factors}

Provide a 2-sentence executive summary of the key risk drivers and their business implications.
Be concise and business-focused."""
)
