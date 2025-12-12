#!/usr/bin/env python3
"""
Update leaderboard.json with new submission scores.

Reads scoring results from scoring_results.json (created by score_new_submission.py)
and updates leaderboard.json with the new scores.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

LEADERBOARD_FILE = Path('leaderboard/leaderboard.json')
SCORING_RESULTS_FILE = Path('scoring_results.json')

def load_leaderboard():
    """Load existing leaderboard or create new one"""
    if LEADERBOARD_FILE.exists():
        try:
            with open(LEADERBOARD_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not read {LEADERBOARD_FILE}: {e}")
            print("   Creating new leaderboard...")
    
    # Create new leaderboard structure
    return {
        "last_updated": datetime.now().isoformat(),
        "submissions": []
    }

def load_scoring_results():
    """Load scoring results from score_new_submission.py"""
    if not SCORING_RESULTS_FILE.exists():
        print(f"❌ ERROR: {SCORING_RESULTS_FILE} not found")
        print("   Run score_new_submission.py first")
        return None
    
    try:
        with open(SCORING_RESULTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Could not read {SCORING_RESULTS_FILE}: {e}")
        return None

def update_leaderboard(leaderboard, scoring_results):
    """Update leaderboard with new scoring results"""
    if not isinstance(scoring_results, list):
        scoring_results = [scoring_results]
    
    updated_count = 0
    new_count = 0
    
    for result in scoring_results:
        submission_name = result.get('submission', 'unknown')
        r2 = result.get('r2', 0.0)
        num_params = result.get('num_params')
        status = result.get('status', 'Failed')
        
        # Find existing entry
        existing_idx = None
        for idx, entry in enumerate(leaderboard['submissions']):
            if entry.get('name') == submission_name:
                existing_idx = idx
                break
        
        # Create new entry
        new_entry = {
            'name': submission_name,
            'r2': r2,
            'params': num_params if num_params is not None else 'N/A',
            'status': status,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'sigma0': result.get('sigma0'),
            'zoff': result.get('zoff')
        }
        
        if existing_idx is not None:
            # Update existing entry
            leaderboard['submissions'][existing_idx] = new_entry
            updated_count += 1
            print(f"✓ Updated: {submission_name} (R² = {r2:.6f})")
        else:
            # Add new entry
            leaderboard['submissions'].append(new_entry)
            new_count += 1
            print(f"✓ Added: {submission_name} (R² = {r2:.6f})")
    
    # Sort by R² descending
    leaderboard['submissions'].sort(key=lambda x: x.get('r2', 0), reverse=True)
    
    # Update timestamp
    leaderboard['last_updated'] = datetime.now().isoformat()
    
    print(f"\n✓ Leaderboard updated: {new_count} new, {updated_count} updated")
    return leaderboard

def save_leaderboard(leaderboard):
    """Save leaderboard to file"""
    # Ensure directory exists
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(LEADERBOARD_FILE, 'w') as f:
            json.dump(leaderboard, f, indent=2)
        print(f"✓ Saved leaderboard to {LEADERBOARD_FILE}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Could not save {LEADERBOARD_FILE}: {e}")
        return False

def main():
    """Main function"""
    print("="*60)
    print("Updating Leaderboard")
    print("="*60)
    
    # Load scoring results
    scoring_results = load_scoring_results()
    if scoring_results is None:
        sys.exit(1)
    
    # Load existing leaderboard
    leaderboard = load_leaderboard()
    
    # Update leaderboard
    leaderboard = update_leaderboard(leaderboard, scoring_results)
    
    # Save leaderboard
    if not save_leaderboard(leaderboard):
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Leaderboard now contains {len(leaderboard['submissions'])} submissions")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
