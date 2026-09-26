import os
import zipfile

def main():
    print("Preparing submission package...")
    
    # Ensure required doc files exist (create minimal versions if missing)
    if not os.path.exists('Documentation_template.md'):
        with open('Documentation_template.md', 'w') as f:
            f.write("# Amazon ML Challenge Documentation\n\nHeuristic Baseline Model.")

    if not os.path.exists('requirements.txt'):
        with open('requirements.txt', 'w') as f:
            f.write("pandas\nnumpy\njellyfish\ntqdm\nscikit-learn\nlightgbm\nscipy\n")

    if not os.path.exists('README.md'):
        with open('README.md', 'w') as f:
            f.write("# Entity Resolution\nRun `python run_fast_submission.py` to generate predictions.\n")

    zip_name = "baseline_submission.zip"
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        print("  Adding output files...")
        zf.write('output/matching_results.tsv', 'output/matching_results.tsv')
        zf.write('output/candidate_pairs.tsv', 'output/candidate_pairs.tsv')
        
        # 2. Source code
        print("  Adding source code...")
        for root, dirs, files in os.walk('src'):
            if '__pycache__' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    # Ensure forward slashes for zip paths
                    zip_path = f"code/business_entity_resolution/{path.replace(chr(92), '/')}"
                    zf.write(path, zip_path)
                    
        # 3. Root code files
        for f in ['main.py', 'run_fast_submission.py', 'requirements.txt', 'README.md']:
            if os.path.exists(f):
                zf.write(f, f"code/business_entity_resolution/{f}")
                
        # 4. Documentation
        print("  Adding documentation...")
        zf.write('Documentation_template.md', 'Documentation_template.md')

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"\nSUCCESS: Created {zip_name} ({size_mb:.2f} MB)")
    print("Ready for upload!")

if __name__ == '__main__':
    main()
