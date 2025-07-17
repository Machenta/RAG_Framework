"""
Demonstration of autocomplete functionality for experiments.

Run this file and then try typing the following in your IDE:

1. experiment. (you should see all available properties)
2. experiment.enabled
3. experiment.name
4. experiment.status
5. experiment.variants
6. experiment.success_metrics

All of these should have full autocomplete support!
"""

from rag_shared.utils.config import Config

def main():
    """Demonstrates autocomplete functionality for experiments."""
    
    # Initialize config
    cfg = Config(
        key_vault_name="RecoveredSpacesKV",
        config_folder="resources/configs",
        config_filename="handbook_config.yml"
    )
    
    print("=== Autocomplete Demo ===\n")
    
    # Method 1: Using cfg.get_experiment() - RECOMMENDED for autocomplete
    experiment = cfg.get_experiment("prompt_optimization")
    
    if experiment:
        print("✅ Method 1: cfg.get_experiment() - Full autocomplete support!")
        print(f"   experiment.enabled = {experiment.enabled}")
        print(f"   experiment.name = {experiment.name}")
        print(f"   experiment.status = {experiment.status}")
        print(f"   experiment.traffic_split = {experiment.traffic_split}")
        print(f"   experiment.variants = {list(experiment.variants.keys())}")
        print(f"   experiment.success_metrics = {experiment.success_metrics}")
        
        # Try changing values - these all have autocomplete!
        experiment.enabled = True
        experiment.status = "active"
        experiment.traffic_split = 50
        
        print(f"\n   After modifications:")
        print(f"   experiment.enabled = {experiment.enabled}")
        print(f"   experiment.status = {experiment.status}")
        print(f"   experiment.traffic_split = {experiment.traffic_split}")
    
    print("\n✅ Method 2: cfg.app.experiments[...] - Also has autocomplete!")
    if cfg.app.experiments:
        exp = cfg.app.experiments["prompt_optimization"]
        print(f"   exp.enabled = {exp.enabled}")
        print(f"   exp.name = {exp.name}")
        # These also have full autocomplete support due to __getitem__ type hints
    
    print("\n✅ Method 3: cfg.app.experiments.get_experiment() - Also has autocomplete!")
    if cfg.app.experiments:
        exp = cfg.app.experiments.get_experiment("prompt_optimization")
        if exp:
            print(f"   exp.enabled = {exp.enabled}")
            print(f"   exp.name = {exp.name}")
    
    print("\n🎉 All methods now provide full IDE autocomplete support!")
    print("\nTry typing any of the following in your IDE:")
    print("  experiment.enabled")
    print("  experiment.name")
    print("  experiment.status")
    print("  experiment.traffic_split")
    print("  experiment.variants")
    print("  experiment.success_metrics")

if __name__ == "__main__":
    main()
