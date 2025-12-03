"""
Visualization script for LLM Council evaluation results
Creates visualizations for Tyson Style scores (accuracy evaluation removed)
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_results(filepath="results/council_evaluation_results.json"):
    """Load evaluation results from JSON"""
    # Check if file exists in results folder first, otherwise try root
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    # Fallback to root directory for backward compatibility
    elif os.path.exists("council_evaluation_results.json"):
        with open("council_evaluation_results.json", 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Results file not found in results/ or current directory")

def create_model_comparison_chart(results):
    """Create bar chart comparing all models' Tyson style scores"""
    # Extract model scores
    model_scores = {}
    
    for eval_data in results['detailed_evaluations']:
        for model_key, model_eval in eval_data['model_evaluations'].items():
            if not model_eval.get('error', False):
                model_name = model_eval['model_name']
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(model_eval['tyson_score'])
    
    # Calculate averages
    avg_scores = {model: np.mean(scores) for model, scores in model_scores.items()}
    
    # Sort by score
    sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    models = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use gradient colors from best to worst
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(models)))
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Tyson Style Score (0-10)', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Tyson Style Scores', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 10)
    
    # Add horizontal lines for score interpretation
    ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Excellent (7+)')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Good (5+)')
    ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Poor (3+)')
    ax.legend(loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/model_comparison_tyson_style.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/model_comparison_tyson_style.png")
    
    return avg_scores

def create_judge_scores_heatmap(results):
    """Create heatmap showing how each judge scored each model"""
    # Prepare data for heatmap
    judge_data = {}
    models_seen = set()
    
    for eval_data in results['detailed_evaluations']:
        for model_key, model_eval in eval_data['model_evaluations'].items():
            if not model_eval.get('error', False):
                model_name = model_eval['model_name']
                models_seen.add(model_name)
                
                if model_name not in judge_data:
                    judge_data[model_name] = {
                        'GPT-4o': [],
                        'Claude-Sonnet-4.5': [],
                        'Gemini-2.5-Pro': [],
                        'DeepSeek-V3': []
                    }
                
                for vote in model_eval['tyson_votes']:
                    judge_name = vote['judge']
                    if judge_name in judge_data[model_name]:
                        judge_data[model_name][judge_name].append(vote['score'])
    
    # Calculate average scores
    heatmap_data = []
    for model in sorted(models_seen):
        row_data = []
        for judge in ['GPT-4o', 'Claude-Sonnet-4.5', 'Gemini-2.5-Pro', 'DeepSeek-V3']:
            if model in judge_data and judge_data[model][judge]:
                avg_score = np.mean(judge_data[model][judge])
                row_data.append(avg_score)
            else:
                row_data.append(np.nan)
        heatmap_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(heatmap_data, 
                      columns=['GPT-4o', 'Claude-4.5', 'Gemini-2.5', 'DeepSeek-V3'],
                      index=sorted(models_seen))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=10, center=5,
                cbar_kws={'label': 'Tyson Style Score'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_title('Judge Scoring Patterns by Model', fontsize=16, fontweight='bold')
    ax.set_xlabel('Judge', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/judge_scores_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/judge_scores_heatmap.png")
    
    return df

def create_judge_consistency_analysis(results):
    """Analyze and visualize judge consistency/variance"""
    judge_variances = {
        'GPT-4o': [],
        'Claude-Sonnet-4.5': [],
        'Gemini-2.5-Pro': [],
        'DeepSeek-V3': []
    }
    
    # Collect score variances for each evaluation
    for eval_data in results['detailed_evaluations']:
        for model_key, model_eval in eval_data['model_evaluations'].items():
            if not model_eval.get('error', False):
                scores_by_judge = {}
                for vote in model_eval['tyson_votes']:
                    judge_name = vote['judge']
                    if judge_name in judge_variances:
                        scores_by_judge[judge_name] = vote['score']
                
                # Calculate variance from mean for each judge
                mean_score = np.mean(list(scores_by_judge.values()))
                for judge, score in scores_by_judge.items():
                    judge_variances[judge].append(abs(score - mean_score))
    
    # Create boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot of deviations - only plot judges with data
    data_to_plot = []
    labels = []
    for judge, variances in judge_variances.items():
        if variances:
            data_to_plot.append(variances)
            labels.append(judge.replace('Sonnet-', '').replace('-Pro', ''))
    
    if data_to_plot:
        bp = ax1.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightsalmon', 'plum']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    ax1.set_xlabel('Judge', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Deviation from Mean Score', fontsize=12, fontweight='bold')
    ax1.set_title('Judge Consistency Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of average variance
    avg_variances = {judge: np.mean(vars) for judge, vars in judge_variances.items() if vars}
    judges = list(avg_variances.keys())
    variances = list(avg_variances.values())
    
    bars = ax2.bar(range(len(judges)), variances, color=colors, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(len(judges)))
    ax2.set_xticklabels([j.replace('Sonnet-', '').replace('-Pro', '') for j in judges])
    ax2.set_xlabel('Judge', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Deviation', fontsize=12, fontweight='bold')
    ax2.set_title('Average Judge Deviation from Consensus', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Judge Agreement Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/judge_consistency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/judge_consistency.png")
    
    return judge_variances

def create_group_performance_radar(results):
    """Create radar chart for group performance if groups exist"""
    if 'group_results' not in results or not results['group_results']:
        print("No group results found - skipping radar chart")
        return None
    
    # Prepare data
    groups = []
    scores = []
    
    for group_name, group_data in results['group_results'].items():
        display_name = {
            'group_A_finetuned_only': 'Fine-tuned Only',
            'group_B_base_rag': 'Base + RAG',
            'group_C_finetuned_rag': 'Fine-tuned + RAG'
        }.get(group_name, group_name)
        groups.append(display_name)
        scores.append(group_data['average_tyson_score'])
    
    # Create simple bar chart for groups
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(groups)]
    bars = ax.bar(groups, scores, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Tyson Style Score (0-10)', fontsize=14, fontweight='bold')
    ax.set_title('Group Performance Comparison\n(Tyson Style Evaluation)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 10)
    
    # Add reference lines
    ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Excellent (7+)')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Good (5+)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/group_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/group_performance.png")
    
    return dict(zip(groups, scores))

def create_detailed_summary_table(results):
    """Create and print detailed summary statistics"""
    print("\n" + "="*80)
    print("DETAILED EVALUATION SUMMARY")
    print("="*80)
    
    # Model performance summary
    model_summaries = []
    
    for eval_data in results['detailed_evaluations']:
        question_snippet = eval_data['question'][:30] + '...' if len(eval_data['question']) > 30 else eval_data['question']
        
        for model_key, model_eval in eval_data['model_evaluations'].items():
            if not model_eval.get('error', False):
                # Calculate judge agreement (standard deviation)
                judge_scores = [v['score'] for v in model_eval['tyson_votes']]
                agreement = np.std(judge_scores) if len(judge_scores) > 1 else 0
                
                model_summaries.append({
                    'Model': model_eval['model_name'],
                    'Question': question_snippet,
                    'Tyson Score': model_eval['tyson_score'],
                    'Judge Agreement (SD)': agreement,
                    'Min Judge Score': min(judge_scores),
                    'Max Judge Score': max(judge_scores)
                })
    
    df = pd.DataFrame(model_summaries)
    
    # Group by model and calculate statistics
    summary = df.groupby('Model').agg({
        'Tyson Score': ['mean', 'std', 'min', 'max'],
        'Judge Agreement (SD)': 'mean'
    }).round(2)
    
    print("\nModel Performance Statistics:")
    print(summary)
    
    # Judge statistics
    judge_stats = {
        'GPT-4o': [],
        'Claude-Sonnet-4.5': [],
        'Gemini-2.5-Pro': [],
        'DeepSeek-V3': []
    }
    
    for eval_data in results['detailed_evaluations']:
        for model_key, model_eval in eval_data['model_evaluations'].items():
            if not model_eval.get('error', False):
                for vote in model_eval['tyson_votes']:
                    if vote['judge'] in judge_stats:
                        judge_stats[vote['judge']].append(vote['score'])
    
    print("\n" + "-"*80)
    print("Judge Scoring Statistics:")
    for judge, scores in judge_stats.items():
        if scores:
            print(f"\n{judge}:")
            print(f"  Mean Score: {np.mean(scores):.2f}")
            print(f"  Std Dev: {np.std(scores):.2f}")
            print(f"  Min/Max: {min(scores):.1f}/{max(scores):.1f}")
    
    return summary

def main():
    """Main function to run all visualizations"""
    print("="*80)
    print("LLM COUNCIL VISUALIZATION - TYSON STYLE FOCUS")
    print("="*80)
    
    print("\nLoading evaluation results...")
    results = load_results()
    
    print(f"Found {len(results.get('detailed_evaluations', []))} evaluations")
    print(f"Models tested: {results['metadata'].get('models_tested', 'Unknown')}")
    print(f"Judges: {results['metadata'].get('council_judges', 'Unknown')}")
    
    print("\nGenerating visualizations...")
    
    # 1. Model comparison chart
    print("\n1. Creating model comparison chart...")
    avg_scores = create_model_comparison_chart(results)
    
    # 2. Judge scoring heatmap
    print("\n2. Creating judge scoring heatmap...")
    heatmap_df = create_judge_scores_heatmap(results)
    
    # 3. Judge consistency analysis
    print("\n3. Analyzing judge consistency...")
    consistency_data = create_judge_consistency_analysis(results)
    
    # 4. Group performance (if applicable)
    print("\n4. Creating group performance visualization...")
    group_scores = create_group_performance_radar(results)
    
    # 5. Detailed summary
    print("\n5. Generating detailed statistics...")
    summary_stats = create_detailed_summary_table(results)
    
    # Final summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    
    print("\nGenerated files in results/ folder:")
    print("  ✓ results/model_comparison_tyson_style.png - Bar chart of all models")
    print("  ✓ results/judge_scores_heatmap.png - Heatmap of judge scoring patterns")
    print("  ✓ results/judge_consistency.png - Judge agreement analysis")
    if group_scores:
        print("  ✓ results/group_performance.png - Group comparison chart")
    
    print("\nKey Findings:")
    if avg_scores:
        best_model = max(avg_scores.items(), key=lambda x: x[1])
        worst_model = min(avg_scores.items(), key=lambda x: x[1])
        print(f"  🥇 Best Model: {best_model[0]} (Score: {best_model[1]:.2f}/10)")
        print(f"  🥉 Lowest Model: {worst_model[0]} (Score: {worst_model[1]:.2f}/10)")
        print(f"  📊 Average Score: {np.mean(list(avg_scores.values())):.2f}/10")
        
        # Check if any model achieved "excellent" status
        excellent_models = [m for m, s in avg_scores.items() if s >= 7]
        if excellent_models:
            print(f"  🌟 Excellent Models (7+): {', '.join(excellent_models)}")
        else:
            print("  ⚠️  No models achieved excellent status (7+)")
    
    plt.show()

if __name__ == "__main__":
    main()