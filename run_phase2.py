"""
Phase 2: Advanced Portfolio Management - Main Execution Script
Optimized to achieve >10% returns with sophisticated ensemble strategies
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase2_pipeline import Phase2Pipeline
import matplotlib.pyplot as plt

def main():
    """
    Main execution function for Phase 2
    """
    print("=" * 80)
    print("🚀 FinRL Phase 2: Advanced Portfolio Management")
    print("🎯 Target: >10% Annual Returns with Multi-Stock Ensemble Strategies")
    print("=" * 80)
    
    # Configuration
    config = {
        'stock_list': None,  # Auto-select top stocks
        'target_return': 0.12,  # 12% target return (higher than 10%)
        'optimization_level': 'standard',  # 'quick', 'standard', 'comprehensive'
        'total_timesteps': 40000,  # Increased for better performance
        'n_trials': 40,  # Optimization trials
        'ensemble_size': 5,  # Number of agents in ensemble
        'initial_amount': 1000000.0  # $1M initial capital
    }
    
    print(f"\n⚙️ Configuration:")
    print(f"   Target Return: {config['target_return']:.1%}")
    print(f"   Optimization Level: {config['optimization_level']}")
    print(f"   Training Timesteps: {config['total_timesteps']:,}")
    print(f"   Optimization Trials: {config['n_trials']}")
    print(f"   Ensemble Size: {config['ensemble_size']}")
    print(f"   Initial Capital: ${config['initial_amount']:,.0f}")
    
    # Initialize Phase 2 pipeline
    pipeline = Phase2Pipeline(
        stock_list=config['stock_list'],
        target_return=config['target_return'],
        model_save_path="phase2_models",
        results_path="phase2_results"
    )
    
    try:
        # Run complete pipeline
        print(f"\n🚀 Starting Phase 2 Pipeline...")
        results = pipeline.run_complete_pipeline(
            optimization_level=config['optimization_level'],
            total_timesteps=config['total_timesteps']
        )
        
        if results.get('success', False):
            print(f"\n🎉 Phase 2 Completed Successfully!")
            print("=" * 60)
            
            # Display final results
            if 'report' in results and 'summary' in results['report']:
                summary = results['report']['summary']
                
                print(f"\n📊 Final Performance Summary:")
                print(f"   Average Return: {summary.get('avg_return', 0):.4f} ({summary.get('avg_return', 0)*100:.2f}%)")
                print(f"   Maximum Return: {summary.get('max_return', 0):.4f} ({summary.get('max_return', 0)*100:.2f}%)")
                print(f"   Average Sharpe Ratio: {summary.get('avg_sharpe_ratio', 0):.4f}")
                print(f"   Maximum Sharpe Ratio: {summary.get('max_sharpe_ratio', 0):.4f}")
                print(f"   Average Drawdown: {summary.get('avg_drawdown', 0):.4f} ({summary.get('avg_drawdown', 0)*100:.2f}%)")
                print(f"   Minimum Drawdown: {summary.get('min_drawdown', 0):.4f} ({summary.get('min_drawdown', 0)*100:.2f}%)")
                
                # Target achievement
                target_achieved = summary.get('target_achieved', False)
                max_return = summary.get('max_return', 0)
                
                print(f"\n🎯 Target Achievement:")
                if target_achieved:
                    print(f"   ✅ TARGET ACHIEVED! Maximum return ({max_return*100:.2f}%) exceeds target ({config['target_return']*100:.1f}%)")
                else:
                    print(f"   ❌ Target not achieved. Maximum return ({max_return*100:.2f}%) below target ({config['target_return']*100:.1f}%)")
                
                # Performance rating
                print(f"\n⭐ Performance Rating:")
                if max_return >= 0.15:  # 15%+
                    print(f"   🌟 EXCELLENT (15%+ returns)")
                elif max_return >= 0.12:  # 12%+
                    print(f"   🏆 VERY GOOD (12%+ returns)")
                elif max_return >= 0.10:  # 10%+
                    print(f"   ✅ GOOD (10%+ returns)")
                elif max_return >= 0.08:  # 8%+
                    print(f"   ⚠️ MODERATE (8%+ returns)")
                else:
                    print(f"   ❌ POOR (<8% returns)")
                
                # Risk assessment
                avg_sharpe = summary.get('avg_sharpe_ratio', 0)
                print(f"\n📈 Risk Assessment:")
                if avg_sharpe >= 2.0:
                    print(f"   🌟 EXCELLENT risk-adjusted returns (Sharpe ≥ 2.0)")
                elif avg_sharpe >= 1.5:
                    print(f"   🏆 VERY GOOD risk-adjusted returns (Sharpe ≥ 1.5)")
                elif avg_sharpe >= 1.0:
                    print(f"   ✅ GOOD risk-adjusted returns (Sharpe ≥ 1.0)")
                elif avg_sharpe >= 0.5:
                    print(f"   ⚠️ MODERATE risk-adjusted returns (Sharpe ≥ 0.5)")
                else:
                    print(f"   ❌ POOR risk-adjusted returns (Sharpe < 0.5)")
            
            # Display detailed results
            if 'evaluation' in results:
                print(f"\n📋 Detailed Results by Method:")
                for method, eval_results in results['evaluation'].items():
                    if 'performance_metrics' in eval_results:
                        metrics = eval_results['performance_metrics']
                        print(f"\n   {method.upper()}:")
                        print(f"     Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)")
                        print(f"     Annualized Return: {metrics.get('annualized_return', 0):.4f} ({metrics.get('annualized_return', 0)*100:.2f}%)")
                        print(f"     Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                        print(f"     Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)")
                        print(f"     Win Rate: {metrics.get('win_rate', 0):.4f} ({metrics.get('win_rate', 0)*100:.1f}%)")
                        print(f"     Total Trades: {metrics.get('total_trades', 0)}")
            
            # Files generated
            print(f"\n📁 Generated Files:")
            print(f"   Models: phase2_models/")
            print(f"   Results: phase2_results/")
            print(f"   Performance Plots: phase2_results/*.png")
            print(f"   CSV Reports: phase2_results/*.csv")
            
            # Next steps
            print(f"\n🚀 Next Steps:")
            if target_achieved:
                print(f"   1. ✅ Target achieved! Consider live trading implementation")
                print(f"   2. 🔧 Fine-tune parameters for even better performance")
                print(f"   3. 📊 Implement real-time monitoring and rebalancing")
                print(f"   4. 🎯 Scale up capital allocation")
            else:
                print(f"   1. 🔧 Try 'comprehensive' optimization level")
                print(f"   2. 📈 Increase training timesteps")
                print(f"   3. 🎯 Experiment with different stock selections")
                print(f"   4. 🤖 Try additional ensemble methods")
            
            print(f"\n💡 Pro Tips for >10% Returns:")
            print(f"   • Use 'comprehensive' optimization for best results")
            print(f"   • Increase training timesteps to 50,000+ for better learning")
            print(f"   • Select high-growth stocks (RELIANCE, TCS, HDFCBANK)")
            print(f"   • Monitor and rebalance portfolio regularly")
            print(f"   • Consider market timing strategies")
            
        else:
            print(f"\n❌ Phase 2 Failed!")
            if 'error' in results:
                print(f"   Error: {results['error']}")
            print(f"\n🔧 Troubleshooting:")
            print(f"   1. Check data availability in processed_data/")
            print(f"   2. Ensure sufficient memory for training")
            print(f"   3. Try 'quick' optimization level first")
            print(f"   4. Reduce total_timesteps if memory issues")
    
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        print(f"\n🔧 Quick Fixes:")
        print(f"   1. Install missing dependencies: pip install -r requirements.txt")
        print(f"   2. Check data files in processed_data/")
        print(f"   3. Ensure sufficient disk space")
        print(f"   4. Try with fewer stocks first")
    
    print(f"\n" + "=" * 80)
    print(f"🏁 Phase 2 Execution Complete!")
    print(f"=" * 80)

def run_quick_test():
    """
    Run a quick test of Phase 2 with minimal resources
    """
    print("🧪 Quick Test Mode - Phase 2")
    print("=" * 40)
    
    # Quick configuration
    config = {
        'stock_list': ["RELIANCE", "TCS", "HDFCBANK"],  # Limited stocks
        'target_return': 0.08,  # Lower target for quick test
        'optimization_level': 'quick',
        'total_timesteps': 5000,  # Minimal training
        'n_trials': 5,  # Minimal optimization
    }
    
    print(f"⚙️ Quick Test Configuration:")
    print(f"   Stocks: {config['stock_list']}")
    print(f"   Target: {config['target_return']:.1%}")
    print(f"   Timesteps: {config['total_timesteps']:,}")
    
    # Initialize pipeline
    pipeline = Phase2Pipeline(
        stock_list=config['stock_list'],
        target_return=config['target_return'],
        model_save_path="quick_test_models",
        results_path="quick_test_results"
    )
    
    try:
        # Run quick test
        results = pipeline.run_complete_pipeline(
            optimization_level=config['optimization_level'],
            total_timesteps=config['total_timesteps']
        )
        
        if results.get('success', False):
            print("✅ Quick test completed successfully!")
            if 'report' in results and 'summary' in results['report']:
                summary = results['report']['summary']
                print(f"   Max Return: {summary.get('max_return', 0)*100:.2f}%")
                print(f"   Max Sharpe: {summary.get('max_sharpe_ratio', 0):.4f}")
        else:
            print("❌ Quick test failed!")
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FinRL Phase 2: Advanced Portfolio Management')
    parser.add_argument('--quick', action='store_true', help='Run quick test mode')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive optimization')
    parser.add_argument('--timesteps', type=int, default=40000, help='Training timesteps')
    parser.add_argument('--target', type=float, default=0.12, help='Target return (default: 0.12)')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        # Update configuration based on arguments
        if args.comprehensive:
            optimization_level = 'comprehensive'
        else:
            optimization_level = 'standard'
        
        # Run main pipeline
        main()
