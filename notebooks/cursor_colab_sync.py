#!/usr/bin/env python3
"""
Cursor AI ↔ Google Colab Integration Helper
Simplifies workflow between local Cursor development and cloud Colab training
"""

import os
import json
import subprocess
import time
from pathlib import Path

class CursorColabSync:
    def __init__(self, repo_path=".", github_repo="rbasina/meetara-lab"):
        self.repo_path = Path(repo_path)
        self.github_repo = github_repo
        self.notebook_path = self.repo_path / "notebooks"
    
    def auto_commit_and_push(self, message="Auto-sync from Cursor AI"):
        """Automatically commit and push changes to GitHub"""
        try:
            # Add all changes
            subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
            
            # Commit with timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            full_message = f"{message} - {timestamp}"
            subprocess.run(["git", "commit", "-m", full_message], cwd=self.repo_path, check=True)
            
            # Push to main
            subprocess.run(["git", "push", "origin", "main"], cwd=self.repo_path, check=True)
            
            print(f"✅ Successfully synced to GitHub: {full_message}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git sync failed: {e}")
            return False
    
    def generate_colab_url(self, notebook_name="colab_gpu_training_template.ipynb"):
        """Generate direct Google Colab URL for notebook"""
        github_path = f"notebooks/{notebook_name}"
        colab_url = f"https://colab.research.google.com/github/{self.github_repo}/blob/main/{github_path}"
        return colab_url
    
    def update_notebook_github_url(self, notebook_name="colab_gpu_training_template.ipynb"):
        """Ensure notebook has correct GitHub URL"""
        notebook_file = self.notebook_path / notebook_name
        
        if not notebook_file.exists():
            print(f"❌ Notebook not found: {notebook_file}")
            return False
        
        try:
            with open(notebook_file, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            
            # Find and update the git clone cell
            for cell in notebook_data.get('cells', []):
                if 'source' in cell:
                    source_lines = cell['source']
                    for i, line in enumerate(source_lines):
                        if 'git clone https://github.com/' in line and 'user/meetara-lab' in line:
                            source_lines[i] = f"!git clone https://github.com/{self.github_repo}.git\n"
                            print(f"✅ Updated GitHub URL in notebook")
            
            # Save updated notebook
            with open(notebook_file, 'w', encoding='utf-8') as f:
                json.dump(notebook_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to update notebook: {e}")
            return False
    
    def create_colab_shortcut(self):
        """Create a quick link file for easy Colab access"""
        shortcut_file = self.repo_path / "OPEN_IN_COLAB.md"
        colab_url = self.generate_colab_url()
        
        content = f"""# 🚀 Quick Colab Access
        
## Open MeeTARA Lab Training in Google Colab:
**[📖 Open Notebook in Colab]({colab_url})**

### Quick Steps:
1. Click the link above
2. **Runtime** → **Change Runtime Type** → **GPU** (T4/V100/A100)
3. **Run All Cells** for automatic 20-100x speed training!

### Current Repository: `{self.github_repo}`
### Last Updated: {time.strftime("%Y-%m-%d %H:%M:%S")}

---
*Auto-generated by Cursor AI ↔ Colab Integration* 🤖
"""
        
        with open(shortcut_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Created Colab shortcut: {shortcut_file}")
        return colab_url

def main():
    """Main integration workflow"""
    print("🚀 Cursor AI ↔ Google Colab Integration")
    print("=" * 50)
    
    sync = CursorColabSync()
    
    # 1. Update notebook with correct GitHub URL
    print("1. Updating notebook GitHub URL...")
    sync.update_notebook_github_url()
    
    # 2. Create quick Colab access link
    print("2. Creating Colab shortcut...")
    colab_url = sync.create_colab_shortcut()
    
    # 3. Auto-commit and push to GitHub
    print("3. Syncing to GitHub...")
    if sync.auto_commit_and_push("Cursor AI: Auto-sync notebook and shortcuts"):
        print(f"\n✅ SUCCESS! Your notebook is ready!")
        print(f"🔗 Open in Colab: {colab_url}")
        print(f"\n💡 Next time, just run: python notebooks/cursor_colab_sync.py")
    else:
        print("❌ Sync failed. Please check Git configuration.")

if __name__ == "__main__":
    main() 
