"""
Command Line Interface for Ouroboros.
"""
import argparse
import sys
from .config import RecursiveCycleConfig, TuningConfig
from .pipeline.self_loop import RecursiveTuningLoop

def main():
    parser = argparse.ArgumentParser(
        description="Ouroboros: Recursive Self-Evolution & Parameter Tuning Engine"
    )
    parser.add_argument(
        "--generations", 
        type=int, 
        default=3, 
        help="Number of recursive evolution generations to execute"
    )
    parser.add_argument(
        "--domain", 
        type=str, 
        default="math_logic", 
        help="Initial focus domain (e.g. math_logic, coding)"
    )
    parser.add_argument(
        "--lora-r", 
        type=int, 
        default=16, 
        help="Rank for low-rank adapter matrices"
    )
    parser.add_argument(
        "--simulate", 
        action="store_true", 
        default=True,
        help="Run in accelerated simulation mode"
    )

    args = parser.parse_args()

    cycle_cfg = RecursiveCycleConfig(
        max_generations=args.generations,
        curriculum_domain=args.domain
    )
    tuning_cfg = TuningConfig(
        lora_r=args.lora_r
    )

    loop = RecursiveTuningLoop(cycle_config=cycle_cfg, tuning_config=tuning_cfg)
    loop.run_cycle()

if __name__ == "__main__":
    main()
