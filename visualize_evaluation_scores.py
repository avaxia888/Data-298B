#!/usr/bin/env python3
"""
Visualize evaluation scores for all models with charts and graphs
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tabulate import tabulate
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load evaluation results"""
    with open("results/evaluation_results.json", "r") as f:
        return json.load(f)

def create_bar_chart(results):
    """Create a bar chart comparing all models' cosine similarity scores"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = []
    scores = []
    colors = []
    categories = []
    
    # Define colors for each category
    color_map = {
        'base_rag': '#2E86AB',      # Blue
        'finetuned': '#A23B72',      # Purple
        'finetuned_rag': '#F18F01'   # Orange
    }
    
    # Collect data and sort by score
    model_data = []
    for model_key, model_info in results["models"].items():
        name = model_info.get("name", model_key)
        score = model_info.get("average_cosine_similarity", 0)
        
        # Determine category
        if "base_rag" in model_key:
            category = 'base_rag'
            display_name = name + " (RAG)"
        elif "finetuned_rag" in model_key:
            category = 'finetuned_rag'
            display_name = name + " (FT+RAG)"
        else:
            category = 'finetuned'
            display_name = name + " (FT)"
        
        model_data.append((display_name, score, category))
    
    # Sort by score descending
    model_data.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted data
    for name, score, category in model_data:
        models.append(name)
        scores.append(score)
        colors.append(color_map[category])
        categories.append(category)
    
    # Create bars
    bars = ax.bar(range(len(models)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Cosine Similarity with Ground Truth', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add horizontal line for average
    overall_avg = np.mean(scores)
    ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.5, label=f'Overall Avg: {overall_avg:.3f}')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#2E86AB', label='Base RAG', alpha=0.8),
        mpatches.Patch(color='#A23B72', label='Finetuned', alpha=0.8),
        mpatches.Patch(color='#F18F01', label='Finetuned + RAG', alpha=0.8),
        mpatches.Patch(color='red', alpha=0.5, label=f'Overall Avg: {overall_avg:.3f}')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Set y-axis limits
    ax.set_ylim(0, max(scores) * 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_category_comparison(results):
    """Create a grouped bar chart comparing categories"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if "summary" in results:
        categories = []
        avg_similarities = []
        avg_times = []
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        category_display = {
            'base_rag': 'Base RAG',
            'finetuned': 'Finetuned',
            'finetuned_rag': 'Finetuned + RAG'
        }
        
        for cat_key in ['base_rag', 'finetuned', 'finetuned_rag']:
            if cat_key in results["summary"]:
                categories.append(category_display[cat_key])
                avg_similarities.append(results["summary"][cat_key]["avg_similarity"])
                avg_times.append(results["summary"][cat_key]["avg_time"] / 1000)  # Convert to seconds
        
        # Similarity comparison
        bars1 = ax1.bar(categories, avg_similarities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Average Cosine Similarity', fontsize=12, fontweight='bold')
        ax1.set_title('Category Performance: Similarity', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, max(avg_similarities) * 1.2)
        
        # Add value labels
        for bar, val in zip(bars1, avg_similarities):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Response time comparison
        bars2 = ax2.bar(categories, avg_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Average Response Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Category Performance: Response Time', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, max(avg_times) * 1.2)
        
        # Add value labels
        for bar, val in zip(bars2, avg_times):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add grid
        ax1.grid(True, alpha=0.3, axis='y')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Category-wise Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_scatter_plot(results):
    """Create a scatter plot of similarity vs response time"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    categories = {'base_rag': [], 'finetuned': [], 'finetuned_rag': []}
    
    for model_key, model_info in results["models"].items():
        name = model_info.get("name", model_key)
        similarity = model_info.get("average_cosine_similarity", 0)
        response_time = model_info.get("average_response_time", 0) / 1000  # Convert to seconds
        
        # Determine category
        if "base_rag" in model_key:
            categories['base_rag'].append((name, similarity, response_time))
        elif "finetuned_rag" in model_key:
            categories['finetuned_rag'].append((name, similarity, response_time))
        else:
            categories['finetuned'].append((name, similarity, response_time))
    
    # Plot each category with different colors and markers
    colors = {'base_rag': '#2E86AB', 'finetuned': '#A23B72', 'finetuned_rag': '#F18F01'}
    markers = {'base_rag': 'o', 'finetuned': 's', 'finetuned_rag': '^'}
    labels = {'base_rag': 'Base RAG', 'finetuned': 'Finetuned', 'finetuned_rag': 'Finetuned + RAG'}
    
    for cat_key, data in categories.items():
        if data:
            names, similarities, times = zip(*data)
            scatter = ax.scatter(times, similarities, 
                               c=colors[cat_key], 
                               marker=markers[cat_key], 
                               s=150, 
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=1.5,
                               label=labels[cat_key])
            
            # Add annotations for each point
            for name, sim, time in data:
                # Only annotate if not too crowded
                ax.annotate(name, 
                          (time, sim),
                          fontsize=8,
                          xytext=(5, 5),
                          textcoords='offset points',
                          alpha=0.7)
    
    # Add quadrant lines
    avg_similarity = np.mean([m["average_cosine_similarity"] for m in results["models"].values()])
    avg_time = np.mean([m["average_response_time"] for m in results["models"].values()]) / 1000
    
    ax.axhline(y=avg_similarity, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=avg_time, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add quadrant labels
    ax.text(0.02, 0.98, 'Fast & Accurate', transform=ax.transAxes, 
            fontsize=10, va='top', alpha=0.5, style='italic')
    ax.text(0.98, 0.98, 'Slow & Accurate', transform=ax.transAxes, 
            fontsize=10, va='top', ha='right', alpha=0.5, style='italic')
    ax.text(0.02, 0.02, 'Fast & Less Accurate', transform=ax.transAxes, 
            fontsize=10, alpha=0.5, style='italic')
    ax.text(0.98, 0.02, 'Slow & Less Accurate', transform=ax.transAxes, 
            fontsize=10, ha='right', alpha=0.5, style='italic')
    
    ax.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Efficiency Trade-off', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_top_bottom_chart(results):
    """Create a chart showing top and bottom performers"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get all models sorted by similarity
    all_models = []
    for model_key, model_info in results["models"].items():
        name = model_info.get("name", model_key)
        similarity = model_info.get("average_cosine_similarity", 0)
        
        # Add category suffix
        if "base_rag" in model_key:
            name += " (RAG)"
        elif "finetuned_rag" in model_key:
            name += " (FT+RAG)"
        else:
            name += " (FT)"
        
        all_models.append((name, similarity))
    
    all_models.sort(key=lambda x: x[1], reverse=True)
    
    # Top 5 performers
    top_5 = all_models[:5]
    top_names, top_scores = zip(*top_5)
    
    bars1 = ax1.barh(range(len(top_names)), top_scores, 
                      color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels(top_names)
    ax1.set_xlabel('Cosine Similarity Score', fontsize=11, fontweight='bold')
    ax1.set_title('Top 5 Performers', fontsize=13, fontweight='bold', color='#2ECC71')
    ax1.set_xlim(0, 0.6)
    
    # Add value labels
    for bar, score in zip(bars1, top_scores):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Bottom 5 performers
    bottom_5 = all_models[-5:]
    bottom_names, bottom_scores = zip(*bottom_5)
    
    bars2 = ax2.barh(range(len(bottom_names)), bottom_scores, 
                      color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(bottom_names)))
    ax2.set_yticklabels(bottom_names)
    ax2.set_xlabel('Cosine Similarity Score', fontsize=11, fontweight='bold')
    ax2.set_title('Bottom 5 Performers', fontsize=13, fontweight='bold', color='#E74C3C')
    ax2.set_xlim(0, 0.6)
    
    # Add value labels
    for bar, score in zip(bars2, bottom_scores):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='x')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Performance Extremes', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def print_summary_table(results):
    """Print a summary table of all results"""
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY TABLE")
    print("="*80)
    
    # Prepare data for table
    table_data = []
    
    for model_key, model_info in results["models"].items():
        name = model_info.get("name", model_key)
        category = model_info.get("category", "")
        similarity = model_info.get("average_cosine_similarity", 0)
        response_time = model_info.get("average_response_time", 0)
        
        # Format category
        if "base_rag" in model_key:
            category = "Base RAG"
        elif "finetuned_rag" in model_key:
            category = "Finetuned + RAG"
        else:
            category = "Finetuned"
        
        table_data.append([
            name,
            category,
            f"{similarity:.3f}",
            f"{response_time:.1f} ms"
        ])
    
    # Sort by similarity score
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    # Add rank
    ranked_data = []
    for i, row in enumerate(table_data, 1):
        ranked_data.append([i] + row)
    
    headers = ["Rank", "Model Name", "Category", "Similarity", "Response Time"]
    print(tabulate(ranked_data, headers=headers, tablefmt="grid"))
    
    # Print category averages
    if "summary" in results:
        print("\n" + "="*80)
        print("CATEGORY AVERAGES")
        print("="*80)
        
        summary_data = []
        for cat_key, cat_data in results["summary"].items():
            cat_name = cat_key.replace("_", " ").upper()
            num_models = len(cat_data.get("models", []))
            avg_sim = cat_data.get("avg_similarity", 0)
            avg_time = cat_data.get("avg_time", 0)
            
            summary_data.append([
                cat_name,
                num_models,
                f"{avg_sim:.3f}",
                f"{avg_time:.1f} ms"
            ])
        
        summary_headers = ["Category", "# Models", "Avg Similarity", "Avg Response Time"]
        print(tabulate(summary_data, headers=summary_headers, tablefmt="grid"))

def main():
    """Main function to create all visualizations"""
    # Load results
    results = load_results()
    
    # Print summary table
    print_summary_table(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Overall bar chart
    fig1 = create_bar_chart(results)
    fig1.savefig('evaluation_scores_bar_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: evaluation_scores_bar_chart.png")
    
    # 2. Category comparison
    fig2 = create_category_comparison(results)
    fig2.savefig('category_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: category_comparison.png")
    
    # 3. Scatter plot
    fig3 = create_scatter_plot(results)
    fig3.savefig('performance_vs_efficiency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_vs_efficiency.png")
    
    # 4. Top/Bottom performers
    fig4 = create_top_bottom_chart(results)
    fig4.savefig('top_bottom_performers.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: top_bottom_performers.png")
    
    # Don't show plots interactively to avoid blocking
    # plt.show()
    
    print("\n" + "="*80)
    print(f"Evaluation timestamp: {results.get('timestamp', 'N/A')}")
    print("All visualizations have been saved and displayed!")
    print("="*80)

if __name__ == "__main__":
    main()