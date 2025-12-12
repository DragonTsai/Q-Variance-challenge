# score_new_submission.py - scores new submission dataset.parquet files
# This script is designed for GitHub Actions to score submissions in PRs
# Fail-safe with error handling - won't crash if submissions are missing files
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, poisson
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import json
try:
    import requests
except ImportError:
    requests = None

def qvar(z, s0, zoff):
    """Q-variance function: σ²(z) = σ₀² + (z - z₀)²/2"""
    return (s0**2 + (z - zoff)**2 / 2)

def find_modified_submissions(pr_number=None):
    """Find which submission folders were modified in the PR or check all folders"""
    submissions_dir = Path('submissions')
    if not submissions_dir.exists():
        return []
    
    # Get all submission folders
    all_folders = [d.name for d in submissions_dir.iterdir() if d.is_dir()]
    
    # If PR number provided, try to use GitHub API to find changed files
    if pr_number:
        try:
            if requests is None:
                raise ImportError("requests not available")
            # Use GitHub API to get PR files (no auth needed for public repos)
            repo = os.environ.get('GITHUB_REPOSITORY', 'q-variance/challenge')
            api_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                files = response.json()
                changed_folders = set()
                for file_info in files:
                    file_path = file_info.get('filename', '')
                    if file_path.startswith('submissions/'):
                        parts = file_path.split('/')
                        if len(parts) >= 2:
                            changed_folders.add(parts[1])
                if changed_folders:
                    # Only score folders that were actually changed
                    return [f for f in all_folders if f in changed_folders]
        except Exception as e:
            print(f"Note: Could not fetch PR files from API: {e}")
            print("   Will check all submission folders instead")
    
    # Fallback: return all folders (will skip ones without dataset.parquet)
    return all_folders

def score_submission(submission_folder):
    """Score a single submission - fail-safe with error handling"""
    submission_path = Path('submissions') / submission_folder
    dataset_path = submission_path / 'dataset.parquet'
    
    print(f"\n{'='*60}")
    print(f"Scoring submission: {submission_folder}")
    print(f"{'='*60}")
    
    # Check if dataset.parquet exists
    if not dataset_path.exists():
        print(f"⚠️  WARNING: dataset.parquet not found in {submission_path}")
        print(f"   Skipping {submission_folder}")
        return None
    
    # Read the submission dataset
    try:
        df = pd.read_parquet(dataset_path)
        print(f"✓ Loaded {len(df)} windows from {dataset_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to read {dataset_path}: {e}")
        return None
    
    # Validate required columns
    required_columns = ['ticker', 'date', 'T', 'z', 'sigma']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ ERROR: Missing required columns: {missing_columns}")
        return None
    
    try:
        data = df.copy()
        data["var"] = data.sigma**2
        
        print(f"   z has NaNs: {data['z'].isna().sum()}")
        
        # Bin the data
        zmax = 0.6
        delz = 0.025*2
        nbins = int(2*zmax/delz + 1)
        bins = np.linspace(-zmax, zmax, nbins)
        
        binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
                       .groupby('z_bin', observed=False)
                       .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
                       .dropna())
        
        if len(binned) == 0:
            print("❌ ERROR: No valid binned data")
            return None
        
        # Fit to q-variance curve (using fixed baseline parameters)
        popt = [0.2586, 0.0214]  # Baseline fit parameters
        
        fitted = qvar(binned.z_mid, popt[0], popt[1])
        r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)
        
        print(f"✓ Q-Variance fit: σ₀ = {popt[0]:.4f}, zoff = {popt[1]:.4f}, R² = {r2:.6f}")
        
        # Try to extract number of parameters from README
        readme_path = submission_path / 'README.md'
        num_params = None
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text()
                # Look for parameter count in README
                import re
                params_match = re.search(r'(\d+)[\s-]*parameter', readme_content, re.IGNORECASE)
                if params_match:
                    num_params = int(params_match.group(1))
            except:
                pass
        
        # Determine status
        status = "Passed" if r2 >= 0.995 else "Failed"
        
        result = {
            'submission': submission_folder,
            'r2': float(r2),
            'sigma0': float(popt[0]),
            'zoff': float(popt[1]),
            'num_windows': len(data),
            'num_params': num_params,
            'status': status
        }
        
        # Output result in JSON format for leaderboard script
        print(f"\n{'='*60}")
        print("SCORING_RESULT:")
        print(json.dumps(result, indent=2))
        print(f"{'='*60}\n")
        
        return result
    except Exception as e:
        print(f"❌ ERROR: Exception during scoring: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function - fail-safe"""
    try:
        # Check if we have a PR number
        if len(sys.argv) > 1:
            pr_number = sys.argv[1]
            print(f"Processing PR #{pr_number}")
        else:
            pr_number = None
            print("No PR number provided, checking all submission folders...")
        
        # Find modified submissions
        submission_folders = find_modified_submissions(pr_number)
        
        if not submission_folders:
            print("⚠️  No submission folders found")
            # Don't fail - just exit gracefully
            print("   Exiting without error (no submissions to score)")
            sys.exit(0)
        
        # Score each submission
        results = []
        for folder in submission_folders:
            result = score_submission(folder)
            if result:
                results.append(result)
        
        if not results:
            print("⚠️  No valid submissions scored")
            print("   (This is OK if submissions don't have dataset.parquet)")
            # Don't fail - just exit gracefully
            sys.exit(0)
        
        # Save results to file for leaderboard script
        results_file = Path('scoring_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved results to {results_file}")
        except Exception as e:
            print(f"⚠️  WARNING: Could not save results file: {e}")
            # Don't fail - results were printed to stdout
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Exit with error code but don't crash the workflow
        sys.exit(1)

if __name__ == "__main__":
    main()
