
import argparse
from subprocess import call


parser = argparse.ArgumentParser(description='Add to existing name')
parser.add_argument('--algo', help='add to existing name especially if I am testing some new feature.')
args = parser.parse_args()
algo = args.algo

call(["python","anomaly_likelihood.py","--algo",algo])
call(["python","run.py","-d",algo,"--optimize","--score","--normalize","--skipConfirmation"])
