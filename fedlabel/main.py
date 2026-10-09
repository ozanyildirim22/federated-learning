import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argparse import ArgumentParser
from fedlabel.config import get_args
from fedlabel.runner import FedLabelRunner

def main():
    parser = ArgumentParser(description="FedLabel Runner")
    args = get_args(parser)
    runner = FedLabelRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
