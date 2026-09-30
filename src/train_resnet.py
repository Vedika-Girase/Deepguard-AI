"""ResNet18 entry point. Use after CNN experiments are finalized."""
import sys
from run_experiment import main
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--model", "resnet18", "--tag", "resnet18_source_independent"] + sys.argv[1:]
    main()
