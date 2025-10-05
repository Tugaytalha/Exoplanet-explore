"""
XGBoost Model Structure Visualization
======================================
This script visualizes the structure of a trained XGBoost model
and saves comprehensive diagrams as image files.

Features:
- Individual decision tree visualization
- Feature importance plots
- Model architecture overview
- Tree depth analysis
- Model complexity metrics

Author: Exoplanet ML Team
"""

import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import os

# Try to import xgboost plotting utilities
try:
    import xgboost as xgb
    from xgboost import plot_tree, plot_importance
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost plotting utilities not available")

# Try to import graphviz for better tree visualization
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Info: Graphviz not available - using matplotlib fallback")


class ModelStructureVisualizer:
    """
    Comprehensive visualization toolkit for XGBoost model structure.
    """
    
    def __init__(self, model_path, features_path=None, output_dir='model_visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the trained model (.joblib file)
            features_path: Path to feature names JSON (optional)
            output_dir: Directory to save visualizations
        """
        self.model_path = model_path
        self.features_path = features_path
        self.output_dir = output_dir
        self.model = None
        self.feature_names = None
        self.booster = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and feature names."""
        print("=" * 70)
        print("LOADING MODEL")
        print("=" * 70)
        
        # Load model
        print(f"\nLoading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(self.model).__name__}")
        
        # Get booster
        if hasattr(self.model, 'get_booster'):
            self.booster = self.model.get_booster()
            print(f"✓ Booster extracted")
        
        # Load feature names
        if self.features_path and os.path.exists(self.features_path):
            print(f"\nLoading feature names from: {self.features_path}")
            with open(self.features_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✓ Feature names loaded: {len(self.feature_names)} features")
        elif hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_.tolist()
            print(f"✓ Feature names extracted from model: {len(self.feature_names)} features")
        else:
            # Create generic feature names
            n_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 0
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
            print(f"⚠ Using generic feature names: {len(self.feature_names)} features")
    
    def print_model_summary(self):
        """Print detailed model statistics."""
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("=" * 70)
        
        # Basic info
        if hasattr(self.model, '__class__'):
            print(f"\nModel Class: {self.model.__class__.__name__}")
        
        # Number of features
        if hasattr(self.model, 'n_features_in_'):
            print(f"Input Features: {self.model.n_features_in_}")
        
        # Number of classes
        if hasattr(self.model, 'n_classes_'):
            print(f"Output Classes: {self.model.n_classes_}")
        
        # Number of trees
        if hasattr(self.model, 'n_estimators'):
            print(f"Number of Trees (Estimators): {self.model.n_estimators}")
        
        # Best iteration
        if hasattr(self.model, 'best_iteration'):
            print(f"Best Iteration: {self.model.best_iteration}")
        
        # Booster info
        if self.booster:
            print(f"\nBooster Information:")
            config = self.booster.save_config()
            import json
            config_dict = json.loads(config)
            
            # Print key parameters
            if 'learner' in config_dict:
                learner = config_dict['learner']
                if 'gradient_booster' in learner:
                    gb = learner['gradient_booster']
                    if 'tree_train_param' in gb:
                        params = gb['tree_train_param']
                        print(f"  Max Depth: {params.get('max_depth', 'N/A')}")
                        print(f"  Learning Rate: {params.get('learning_rate', 'N/A')}")
                        print(f"  Min Child Weight: {params.get('min_child_weight', 'N/A')}")
                        print(f"  Gamma: {params.get('gamma', 'N/A')}")
                        print(f"  Subsample: {params.get('subsample', 'N/A')}")
                        print(f"  Colsample by Tree: {params.get('colsample_bytree', 'N/A')}")
        
        # Feature importance stats
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            print(f"\nFeature Importance Statistics:")
            print(f"  Top feature importance: {np.max(importances):.6f}")
            print(f"  Mean importance: {np.mean(importances):.6f}")
            print(f"  Features with importance > 0.01: {np.sum(importances > 0.01)}")
            print(f"  Features with importance = 0: {np.sum(importances == 0)}")
    
    def visualize_model_architecture(self):
        """Create a high-level architecture diagram of the model."""
        print("\n" + "=" * 70)
        print("CREATING MODEL ARCHITECTURE DIAGRAM")
        print("=" * 70)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Define positions for architecture components
        y_positions = {
            'input': 0.85,
            'imputation': 0.70,
            'scaling': 0.55,
            'xgboost': 0.30,
            'output': 0.05
        }
        
        # Color scheme
        colors = {
            'input': '#3498db',
            'preprocessing': '#9b59b6',
            'model': '#e74c3c',
            'output': '#2ecc71'
        }
        
        # Input layer
        n_features = len(self.feature_names) if self.feature_names else self.model.n_features_in_
        input_box = plt.Rectangle((0.25, y_positions['input']-0.05), 0.5, 0.08, 
                                   facecolor=colors['input'], alpha=0.3, edgecolor=colors['input'], linewidth=2)
        ax.add_patch(input_box)
        ax.text(0.5, y_positions['input'], f'Input Layer\n{n_features} Features', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Imputation
        impute_box = plt.Rectangle((0.25, y_positions['imputation']-0.05), 0.5, 0.08,
                                    facecolor=colors['preprocessing'], alpha=0.3, 
                                    edgecolor=colors['preprocessing'], linewidth=2)
        ax.add_patch(impute_box)
        ax.text(0.5, y_positions['imputation'], 'Missing Value Imputation\n(Median Strategy)', 
                ha='center', va='center', fontsize=11)
        
        # Scaling
        scale_box = plt.Rectangle((0.25, y_positions['scaling']-0.05), 0.5, 0.08,
                                   facecolor=colors['preprocessing'], alpha=0.3, 
                                   edgecolor=colors['preprocessing'], linewidth=2)
        ax.add_patch(scale_box)
        ax.text(0.5, y_positions['scaling'], 'Feature Scaling\n(StandardScaler)', 
                ha='center', va='center', fontsize=11)
        
        # XGBoost ensemble
        n_trees = self.model.n_estimators if hasattr(self.model, 'n_estimators') else 'N/A'
        n_classes = self.model.n_classes_ if hasattr(self.model, 'n_classes_') else 'N/A'
        max_depth = self.model.max_depth if hasattr(self.model, 'max_depth') else 'N/A'
        
        xgb_box = plt.Rectangle((0.15, y_positions['xgboost']-0.1), 0.7, 0.18,
                                 facecolor=colors['model'], alpha=0.3, 
                                 edgecolor=colors['model'], linewidth=3)
        ax.add_patch(xgb_box)
        ax.text(0.5, y_positions['xgboost']+0.05, 'XGBoost Ensemble', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, y_positions['xgboost']-0.02, 
                f'{n_trees} Decision Trees | Max Depth: {max_depth} | {n_classes} Classes',
                ha='center', va='center', fontsize=10)
        
        # Show a few example trees
        tree_width = 0.08
        tree_spacing = 0.12
        start_x = 0.5 - (min(5, n_trees if isinstance(n_trees, int) else 5) * tree_spacing) / 2
        for i in range(min(5, n_trees if isinstance(n_trees, int) else 5)):
            tree_x = start_x + i * tree_spacing
            tree_box = plt.Rectangle((tree_x, y_positions['xgboost']-0.08), tree_width, 0.06,
                                      facecolor='white', edgecolor=colors['model'], linewidth=1)
            ax.add_patch(tree_box)
            ax.text(tree_x + tree_width/2, y_positions['xgboost']-0.05, f'T{i+1}',
                    ha='center', va='center', fontsize=8)
        
        if isinstance(n_trees, int) and n_trees > 5:
            ax.text(0.5, y_positions['xgboost']-0.11, '... and more trees',
                    ha='center', va='center', fontsize=8, style='italic')
        
        # Output layer
        output_box = plt.Rectangle((0.25, y_positions['output']-0.05), 0.5, 0.08,
                                    facecolor=colors['output'], alpha=0.3, 
                                    edgecolor=colors['output'], linewidth=2)
        ax.add_patch(output_box)
        
        # Get class names if available
        output_text = f'Output Layer\n{n_classes} Classes: '
        if hasattr(self.model, 'classes_'):
            class_names = ', '.join(map(str, self.model.classes_))
            output_text += f'{class_names}'
        else:
            output_text += 'CANDIDATE, CONFIRMED, FALSE POSITIVE'
        
        ax.text(0.5, y_positions['output'], output_text,
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Draw arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black', alpha=0.6)
        ax.annotate('', xy=(0.5, y_positions['imputation']+0.03), 
                   xytext=(0.5, y_positions['input']-0.05),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, y_positions['scaling']+0.03), 
                   xytext=(0.5, y_positions['imputation']-0.05),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, y_positions['xgboost']+0.08), 
                   xytext=(0.5, y_positions['scaling']-0.05),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(0.5, y_positions['output']+0.03), 
                   xytext=(0.5, y_positions['xgboost']-0.1),
                   arrowprops=arrow_props)
        
        # Add title
        plt.title('XGBoost Model Architecture\nKOI Disposition Prediction Pipeline', 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.98, f'Generated: {timestamp}', 
                ha='center', va='top', fontsize=8, transform=ax.transAxes, style='italic')
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Save
        output_path = os.path.join(self.output_dir, 'model_architecture.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Architecture diagram saved to: {output_path}")
        plt.close()
    
    def visualize_feature_importance(self, top_n=30):
        """Create enhanced feature importance visualization."""
        print("\n" + "=" * 70)
        print(f"CREATING FEATURE IMPORTANCE VISUALIZATION (Top {top_n})")
        print("=" * 70)
        
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠ Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': self.feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Get top N
        top_features = feature_df.head(top_n)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # Plot 1: Horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=10)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
        ax1.set_title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax1.text(row['importance'], i, f" {row['importance']:.4f}", 
                    va='center', fontsize=9)
        
        # Plot 2: Cumulative importance
        cumsum = feature_df['importance'].cumsum()
        ax2.plot(range(1, len(cumsum)+1), cumsum, linewidth=2, color='#e74c3c')
        ax2.fill_between(range(1, len(cumsum)+1), cumsum, alpha=0.3, color='#e74c3c')
        ax2.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cumulative Importance', fontsize=14, fontweight='bold')
        ax2.set_title('Cumulative Feature Importance', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add markers for important thresholds
        thresholds = [0.5, 0.75, 0.9, 0.95]
        for threshold in thresholds:
            if cumsum.max() >= threshold:
                n_features = (cumsum >= threshold).idxmax() + 1
                ax2.axhline(threshold, color='gray', linestyle='--', alpha=0.5)
                ax2.axvline(n_features, color='gray', linestyle='--', alpha=0.5)
                ax2.text(n_features, threshold, f'  {int(threshold*100)}% at {n_features} features', 
                        fontsize=9, va='bottom')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'feature_importance_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Feature importance visualization saved to: {output_path}")
        plt.close()
        
        # Also create XGBoost native importance plot
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            xgb.plot_importance(self.model, ax=ax, max_num_features=top_n, 
                               importance_type='weight', show_values=True)
            ax.set_title(f'XGBoost Feature Importance (Weight) - Top {top_n}', 
                        fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, 'xgboost_native_importance.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ XGBoost native importance plot saved to: {output_path}")
            plt.close()
        except Exception as e:
            print(f"⚠ Could not create XGBoost native plot: {e}")
    
    def visualize_individual_trees(self, tree_indices=[0, 1, 2], max_depth=3):
        """Visualize specific trees from the ensemble."""
        print("\n" + "=" * 70)
        print(f"CREATING INDIVIDUAL TREE VISUALIZATIONS")
        print("=" * 70)
        
        if not XGB_AVAILABLE:
            print("⚠ XGBoost not available for tree visualization")
            return
        
        for tree_idx in tree_indices:
            try:
                fig, ax = plt.subplots(figsize=(20, 12))
                xgb.plot_tree(self.model, num_trees=tree_idx, ax=ax)
                ax.set_title(f'Decision Tree #{tree_idx} Structure', 
                           fontsize=16, fontweight='bold', pad=20)
                
                output_path = os.path.join(self.output_dir, f'tree_{tree_idx}_structure.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Tree {tree_idx} visualization saved to: {output_path}")
                plt.close()
            except Exception as e:
                print(f"⚠ Could not visualize tree {tree_idx}: {e}")
    
    def visualize_tree_depth_analysis(self):
        """Analyze and visualize tree depth distribution."""
        print("\n" + "=" * 70)
        print("CREATING TREE DEPTH ANALYSIS")
        print("=" * 70)
        
        if not self.booster:
            print("⚠ Booster not available for depth analysis")
            return
        
        try:
            # Get tree dataframes
            tree_df = self.booster.trees_to_dataframe()
            
            # Analyze depths per tree
            tree_depths = []
            for tree_id in tree_df['Tree'].unique():
                tree_data = tree_df[tree_df['Tree'] == tree_id]
                # Depth is the maximum depth value in the tree
                max_depth = tree_data['Depth'].max()
                tree_depths.append(max_depth)
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Depth distribution histogram
            ax1.hist(tree_depths, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Tree Depth', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Trees', fontsize=12, fontweight='bold')
            ax1.set_title('Distribution of Tree Depths', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            ax1.axvline(np.mean(tree_depths), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(tree_depths):.2f}')
            ax1.legend()
            
            # Plot 2: Depth over tree index
            ax2.plot(tree_depths, linewidth=1, alpha=0.7, color='#2ecc71')
            ax2.set_xlabel('Tree Index', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Depth', fontsize=12, fontweight='bold')
            ax2.set_title('Tree Depth vs Tree Index', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Node count per tree
            node_counts = tree_df.groupby('Tree').size()
            ax3.hist(node_counts, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Number of Trees', fontsize=12, fontweight='bold')
            ax3.set_title('Distribution of Node Counts', fontsize=14, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            # Plot 4: Statistics table
            ax4.axis('off')
            stats_data = [
                ['Metric', 'Value'],
                ['Total Trees', f'{len(tree_depths)}'],
                ['Mean Depth', f'{np.mean(tree_depths):.2f}'],
                ['Max Depth', f'{np.max(tree_depths)}'],
                ['Min Depth', f'{np.min(tree_depths)}'],
                ['Std Depth', f'{np.std(tree_depths):.2f}'],
                ['Total Nodes', f'{len(tree_df)}'],
                ['Mean Nodes/Tree', f'{node_counts.mean():.2f}'],
                ['Leaf Nodes', f'{len(tree_df[tree_df["Feature"] == "Leaf"])}'],
                ['Split Nodes', f'{len(tree_df[tree_df["Feature"] != "Leaf"])}'],
            ]
            
            table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                             colWidths=[0.5, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
            
            # Style header row
            for i in range(2):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(stats_data)):
                for j in range(2):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#ecf0f1')
            
            ax4.set_title('Model Structure Statistics', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save
            output_path = os.path.join(self.output_dir, 'tree_depth_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Tree depth analysis saved to: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"⚠ Could not create depth analysis: {e}")
    
    def create_all_visualizations(self):
        """Create all available visualizations."""
        print("\n" + "=" * 70)
        print(" XGBOOST MODEL STRUCTURE VISUALIZATION")
        print("=" * 70)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        
        # Print model summary
        self.print_model_summary()
        
        # Create visualizations
        self.visualize_model_architecture()
        self.visualize_feature_importance(top_n=30)
        self.visualize_individual_trees(tree_indices=[0, 1, 2])
        self.visualize_tree_depth_analysis()
        
        # Final summary
        print("\n" + "=" * 70)
        print(" VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"\n✓ All visualizations saved to: {self.output_dir}/")
        print(f"\nGenerated files:")
        for file in sorted(os.listdir(self.output_dir)):
            file_path = os.path.join(self.output_dir, file)
            if file.endswith('.png'):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


def find_latest_model(model_dir='model_outputs'):
    """Find the most recent model file."""
    model_files = list(Path(model_dir).glob('xgboost_koi_disposition_*.joblib'))
    if not model_files:
        return None
    # Sort by modification time, get the most recent
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def find_latest_features(model_dir='model_outputs'):
    """Find the most recent feature names file."""
    feature_files = list(Path(model_dir).glob('feature_names_*.json'))
    if not feature_files:
        return None
    latest = max(feature_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Visualize XGBoost model structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest model automatically
  python visualize_model_structure.py
  
  # Specify model path
  python visualize_model_structure.py --model model_outputs/xgboost_koi_disposition_20251004_221246.joblib
  
  # Specify model and features
  python visualize_model_structure.py \\
      --model model_outputs/xgboost_koi_disposition_20251004_221246.joblib \\
      --features model_outputs/feature_names_20251004_221246.json
  
  # Custom output directory
  python visualize_model_structure.py --output my_visualizations/
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (.joblib file). If not specified, uses latest.')
    parser.add_argument('--features', type=str, default=None,
                       help='Path to feature names JSON. If not specified, uses latest or extracts from model.')
    parser.add_argument('--output', type=str, default='model_visualizations',
                       help='Output directory for visualizations (default: model_visualizations)')
    
    args = parser.parse_args()
    
    # Find model if not specified
    model_path = args.model
    if model_path is None:
        print("No model specified, searching for latest model...")
        model_path = find_latest_model()
        if model_path is None:
            print("Error: No trained models found in model_outputs/")
            print("Please train a model first using: python train_koi_disposition.py")
            return
        print(f"Found latest model: {model_path}")
    
    # Find features if not specified
    features_path = args.features
    if features_path is None:
        features_path = find_latest_features()
        if features_path:
            print(f"Found latest features: {features_path}")
    
    # Create visualizer
    visualizer = ModelStructureVisualizer(
        model_path=model_path,
        features_path=features_path,
        output_dir=args.output
    )
    
    # Create all visualizations
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()

