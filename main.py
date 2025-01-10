import argparse

from src.pipeline import run_pipeline
from src.analysis.analysis import analyze_results

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run pipeline and/or analysis with optional parameters')
    parser.add_argument('p', '--pipeline', action='store_true', help='Run ML pipeline')
    parser.add_argument('-a', '--analysis', action='store_true', help='Run statistical analyses')
    parser.add_argument('-rp', '--results_pattern', type=str, default=None, help='Custom suffix of results file to load results from during analysis.')
    parser.add_argument('-ff', '--fetch_feat_imps', action='store_true', help='Fetch feature importance statistics stored in file from previous analysis run (must be False to (over)write the file)')
    parser.add_argument('--plot', action='store_true', help='Plot results, given analysis flag is set.')
    # For dummy data
    parser.add_argument('--n', type=int, default=0, help='Number of participants for dummy data')
    parser.add_argument('--drug_sens', type=float, default=0.5, help='Drug sensitivity for dummy data')
    parser.add_argument('--sex_sens', type=float, default=0.1, help='Sex sensitivity for dummy data')
    parser.add_argument('--age_sens', type=float, default=0.1, help='Age sensitivity for dummy data')
    parser.add_argument('--gen_data', action='store_true', help='Generate synthetic dummy data')
    
    args = parser.parse_args()
    
    return {
        'n': args.n,
        'drug_sens': args.drug_sens,
        'sex_sens': args.sex_sens,
        'age_sens': args.age_sens,
        'gen_data': args.gen_data,
        'pipeline': args.pipeline,
        'analysis': args.analysis,
        'results_pattern': args.results_pattern,
        'fetch_feat_imps': args.fetch_feat_imps,
        'plot': args.plot,
    }

def main():
    args = parse_arguments()
    if args['pipeline']:
        run_pipeline(args)
    if args['analysis']:
        analyze_results(args)

if __name__ == "__main__":
    main()