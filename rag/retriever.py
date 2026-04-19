"""
RAG Retriever - Retrieves similar churn cases from vector store.
"""

import logging
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves similar customer cases using embeddings and vector search.
    
    Pipeline:
    1. Customer input converted to embedding
    2. Similar historical cases searched in vector store
    3. Top matches returned with metadata
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator, 
                 vector_store: VectorStore):
        """
        Initialize RAG retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
        """
        self.embedding_gen = embedding_generator
        self.vector_store = vector_store
    
    def build_knowledge_base(self, training_df: pd.DataFrame,
                           predictions: np.ndarray,
                           probabilities: np.ndarray,
                           sample_size: Optional[int] = None) -> None:
        """
        Build knowledge base from historical customer data.
        
        Args:
            training_df: Training DataFrame with customer features
            predictions: Model predictions (0 or 1)
            probabilities: Prediction probabilities
            sample_size: Optional limit on samples to use
        """
        logger.info("Building RAG knowledge base")
        
        if sample_size:
            indices = np.random.choice(len(training_df), 
                                      min(sample_size, len(training_df)), 
                                      replace=False)
            training_df = training_df.iloc[indices].reset_index(drop=True)
            predictions = predictions[indices]
            probabilities = probabilities[indices]
        
        # Prepare cases
        cases = []
        embeddings_list = []
        
        for idx, (_, row) in enumerate(training_df.iterrows()):
            # Convert row to dict
            customer_dict = row.to_dict()
            
            # Generate embedding
            embedding = self.embedding_gen.encode_customer_profile(customer_dict)
            embeddings_list.append(embedding)
            
            # Create case metadata
            churn_outcome = predictions[idx] == 1
            
            # Handle both 1D and 2D probability arrays
            if probabilities.ndim > 1:
                churn_prob = probabilities[idx][1]
            else:
                churn_prob = probabilities[idx]
            
            case = {
                "customer_features": customer_dict,
                "churn_outcome": churn_outcome,
                "churn_probability": float(churn_prob),
                "retention_strategies": self._get_strategies_for_case(
                    customer_dict, churn_outcome
                ),
                "case_index": idx,
            }
            
            cases.append(case)
        
        # Add to vector store
        embeddings_array = np.array(embeddings_list)
        self.vector_store.add_embeddings(embeddings_array, cases)
        
        logger.info(f"Built KB with {len(cases)} cases")
    
    def retrieve_similar_cases(self, customer_data: Dict[str, Any],
                              k: int = 5,
                              same_outcome_only: bool = False
                              ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve similar historical cases for a customer.
        
        Args:
            customer_data: Customer information
            k: Number of cases to retrieve
            same_outcome_only: Filter to same churn outcome
            
        Returns:
            List of (case, similarity) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_gen.encode_customer_profile(customer_data)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k*2)
        
        # Filter if needed
        if same_outcome_only:
            # User's expected churn outcome not known here, just return top results
            pass
        
        # Return top k
        return results[:k]
    
    def get_case_description(self, case: Dict[str, Any]) -> str:
        """
        Create natural language description of a case.
        
        Args:
            case: Case dictionary
            
        Returns:
            Human-readable case description
        """
        features = case.get("customer_features", {})
        outcome = case.get("churn_outcome", False)
        prob = case.get("churn_probability", 0)
        
        desc_parts = []
        
        # Customer profile
        if "Contract" in features and "InternetService" in features:
            desc_parts.append(
                f"Customer with {features['Contract']} contract and "
                f"{features['InternetService']} service"
            )
        
        # Tenure and charges
        if "tenure" in features:
            desc_parts.append(f"Tenure: {features['tenure']} months")
        
        if "MonthlyCharges" in features:
            desc_parts.append(f"Monthly charge: ${features['MonthlyCharges']:.2f}")
        
        # Outcome
        outcome_text = "CHURNED" if outcome else "RETAINED"
        desc_parts.append(f"Outcome: {outcome_text} (probability: {prob:.1%})")
        
        # Strategies
        strategies = case.get("retention_strategies", [])
        if strategies:
            desc_parts.append(f"Strategy used: {strategies[0]}")
        
        return " | ".join(desc_parts)
    
    def _get_strategies_for_case(self, customer_data: Dict[str, Any],
                                 churned: bool) -> List[str]:
        """
        Generate retention strategies based on customer profile.
        
        Args:
            customer_data: Customer information
            churned: Whether customer churned
            
        Returns:
            List of strategy suggestions
        """
        strategies = []
        
        if not churned:
            # Customer retained - return what kept them
            if customer_data.get("Contract") == "Two year":
                strategies.append("Long-term contract commitment strategy")
            if customer_data.get("TechSupport") == "Yes":
                strategies.append("Premium support retention")
            if customer_data.get("InternetService") == "No":
                strategies.append("Offline service loyalty")
        else:
            # Customer churned - suggest strategy
            if customer_data.get("Contract") == "Month-to-month":
                strategies.append("Offer discounted 1-year contract")
            if customer_data.get("MonthlyCharges", 0) > 80:
                strategies.append("Present cost optimization options")
            if customer_data.get("TechSupport") == "No":
                strategies.append("Offer free tech support trial")
            if customer_data.get("InternetService") == "Fiber optic":
                strategies.append("Highlight speed/performance benefits")
        
        if not strategies:
            strategies.append("Personalized loyalty offer")
        
        return strategies
    
    def format_retrieval_context(self, similar_cases: List[Tuple[Dict, float]]
                                ) -> str:
        """
        Format retrieved cases into context for LLM.
        
        Args:
            similar_cases: List of (case, similarity) tuples
            
        Returns:
            Formatted context string
        """
        context_lines = ["## Similar Historical Cases\n"]
        
        for i, (case, similarity) in enumerate(similar_cases, 1):
            context_lines.append(f"### Case {i} (Similarity: {similarity:.2%})")
            context_lines.append(self.get_case_description(case))
            context_lines.append("")
        
        return "\n".join(context_lines)
