from typing import Dict, List, Optional
import numpy as np

class EvaluationMetrics:
    """Calculate and aggregate evaluation metrics."""
    
    @staticmethod
    def calculate_overall_score(scores: Dict[str, float]) -> float:
        """Calculate weighted overall score from individual scores.
        
        Args:
            scores: Dictionary of individual scores
            
        Returns:
            Weighted overall score
        """
        if not scores:
            return 0.0
            
        # Define weights for different criteria (can be adjusted)
        weights = {
            "style_authenticity": 0.40,  # Most important - Neil's authentic style
            "scientific_accuracy": 0.30,
            "relevance": 0.20,
            "clarity_accessibility": 0.05,
            "engagement": 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, score in scores.items():
            weight = weights.get(key, 0.1)  # Default weight if key not found
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return np.mean(list(scores.values()))
    
    @staticmethod
    def aggregate_batch_scores(
        results: List[Dict[str, float]]
    ) -> Dict[str, any]:
        """Aggregate scores from multiple evaluations.
        
        Args:
            results: List of evaluation score dictionaries
            
        Returns:
            Aggregated statistics
        """
        if not results:
            return {
                "mean": {},
                "std": {},
                "min": {},
                "max": {},
                "count": 0
            }
        
        # Collect all score keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        aggregated = {
            "mean": {},
            "std": {},
            "min": {},
            "max": {},
            "count": len(results)
        }
        
        for key in all_keys:
            values = [r.get(key, 0) for r in results if key in r]
            if values:
                aggregated["mean"][key] = np.mean(values)
                aggregated["std"][key] = np.std(values)
                aggregated["min"][key] = np.min(values)
                aggregated["max"][key] = np.max(values)
        
        return aggregated
    
    @staticmethod
    def calculate_improvement_score(
        before_scores: Dict[str, float],
        after_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement between two evaluations.
        
        Args:
            before_scores: Scores before improvement
            after_scores: Scores after improvement
            
        Returns:
            Improvement percentages for each metric
        """
        improvements = {}
        
        for key in before_scores.keys():
            before = before_scores.get(key, 0)
            after = after_scores.get(key, 0)
            
            if before > 0:
                improvement = ((after - before) / before) * 100
            else:
                improvement = 100 if after > 0 else 0
                
            improvements[key] = improvement
        
        return improvements
    
    @staticmethod
    def get_performance_category(score: float) -> str:
        """Categorize performance based on score.
        
        Args:
            score: Numerical score (0-10 scale)
            
        Returns:
            Performance category string
        """
        if score >= 9:
            return "Excellent"
        elif score >= 7:
            return "Good"
        elif score >= 5:
            return "Fair"
        elif score >= 3:
            return "Poor"
        else:
            return "Very Poor"
    
    @staticmethod
    def format_score_display(
        scores: Dict[str, float],
        include_categories: bool = True
    ) -> str:
        """Format scores for display.
        
        Args:
            scores: Dictionary of scores
            include_categories: Whether to include performance categories
            
        Returns:
            Formatted string for display
        """
        if not scores:
            return "No scores available"
        
        lines = []
        for key, value in scores.items():
            formatted_key = key.replace("_", " ").title()
            if include_categories:
                category = EvaluationMetrics.get_performance_category(value)
                lines.append(f"{formatted_key}: {value:.2f}/10 ({category})")
            else:
                lines.append(f"{formatted_key}: {value:.2f}/10")
        
        return "\n".join(lines)
    
    @staticmethod
    def calculate_consistency_score(
        evaluations: List[Dict[str, float]]
    ) -> float:
        """Calculate consistency score across multiple evaluations.
        
        Args:
            evaluations: List of evaluation score dictionaries
            
        Returns:
            Consistency score (0-1, where 1 is perfectly consistent)
        """
        if len(evaluations) < 2:
            return 1.0
        
        # Calculate coefficient of variation for each metric
        cvs = []
        for key in evaluations[0].keys():
            values = [e.get(key, 0) for e in evaluations]
            mean = np.mean(values)
            std = np.std(values)
            
            if mean > 0:
                cv = std / mean
                cvs.append(cv)
        
        if cvs:
            # Convert CV to consistency score (lower CV = higher consistency)
            avg_cv = np.mean(cvs)
            consistency = max(0, 1 - avg_cv)
            return consistency
        
        return 1.0