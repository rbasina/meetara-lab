#!/usr/bin/env python3
"""
Model Cache Tracker for MeeTARA Lab
Tracks model downloads, caching behavior, and storage locations during Colab sessions
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import psutil

class ModelCacheTracker:
    """Track model downloads and caching behavior"""
    
    def __init__(self):
        self.download_log = []
        self.cache_hits = {}
        self.storage_locations = {}
        self.session_start = time.time()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelCacheTracker")
        
        # Track cache directory
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        self.logger.info(f"📁 Monitoring cache directory: {self.cache_dir}")
    
    def log_download(self, model_name: str, domain: str, download_time: float, 
                    cache_hit: bool, storage_path: str, model_size_mb: float):
        """Log a model download event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "domain": domain,
            "download_time": download_time,
            "cache_hit": cache_hit,
            "storage_path": storage_path,
            "model_size_mb": model_size_mb,
            "session_time": time.time() - self.session_start
        }
        
        self.download_log.append(event)
        
        # Track cache hits
        if cache_hit:
            self.cache_hits[model_name] = self.cache_hits.get(model_name, 0) + 1
        
        # Track storage locations
        self.storage_locations[model_name] = storage_path
        
        self.logger.info(f"📥 {model_name} for {domain}: {'CACHE HIT' if cache_hit else 'DOWNLOADED'} in {download_time:.2f}s")
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        total_downloads = len(self.download_log)
        cache_hits = len([e for e in self.download_log if e["cache_hit"]])
        cache_misses = total_downloads - cache_hits
        total_download_time = sum(e["download_time"] for e in self.download_log if not e["cache_hit"])
        total_model_size = sum(e["model_size_mb"] for e in self.download_log if e["model_size_mb"] != "unknown")
        
        return {
            "total_downloads": total_downloads,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": (cache_hits / total_downloads * 100) if total_downloads > 0 else 0,
            "total_download_time": total_download_time,
            "total_model_size_mb": total_model_size,
            "unique_models": len(set(e["model_name"] for e in self.download_log)),
            "session_duration": time.time() - self.session_start
        }
    
    def get_model_reuse_stats(self) -> Dict:
        """Get statistics on model reuse"""
        model_counts = {}
        for event in self.download_log:
            model_name = event["model_name"]
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        reuse_stats = {}
        for model_name, count in model_counts.items():
            reuse_stats[model_name] = {
                "downloads": count,
                "cache_hits": self.cache_hits.get(model_name, 0),
                "efficiency": (self.cache_hits.get(model_name, 0) / count * 100) if count > 0 else 0
            }
        
        return reuse_stats
    
    def scan_cache_directory(self) -> Dict:
        """Scan the cache directory for existing models"""
        cache_info = {}
        
        if not os.path.exists(self.cache_dir):
            return cache_info
        
        for root, dirs, files in os.walk(self.cache_dir):
            # Look for model files
            model_files = [f for f in files if f.endswith(('.bin', '.safetensors', '.json'))]
            if model_files:
                # Extract model name from path
                path_parts = root.split(os.sep)
                for i, part in enumerate(path_parts):
                    if part == "hub":
                        if i + 1 < len(path_parts):
                            model_name = path_parts[i + 1].replace("--", "/")
                            cache_info[model_name] = {
                                "path": root,
                                "files": len(model_files),
                                "size_mb": sum(os.path.getsize(os.path.join(root, f)) for f in model_files) / (1024 * 1024)
                            }
                        break
        
        return cache_info
    
    def generate_report(self) -> str:
        """Generate a comprehensive report"""
        stats = self.get_cache_stats()
        reuse_stats = self.get_model_reuse_stats()
        cache_scan = self.scan_cache_directory()
        
        report = []
        report.append("=" * 80)
        report.append("📊 MEE TARA LAB - MODEL CACHE TRACKER REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⏱️ Session Duration: {stats['session_duration']/60:.1f} minutes")
        report.append("")
        
        # Overall Statistics
        report.append("📈 OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Downloads: {stats['total_downloads']}")
        report.append(f"Cache Hits: {stats['cache_hits']}")
        report.append(f"Cache Misses: {stats['cache_misses']}")
        report.append(f"Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
        report.append(f"Total Download Time: {stats['total_download_time']/60:.1f} minutes")
        report.append(f"Total Model Size: {stats['total_model_size_mb']:.1f} MB")
        report.append(f"Unique Models: {stats['unique_models']}")
        report.append("")
        
        # Model Reuse Statistics
        report.append("🔄 MODEL REUSE STATISTICS")
        report.append("-" * 40)
        for model_name, info in reuse_stats.items():
            report.append(f"{model_name}:")
            report.append(f"  Downloads: {info['downloads']}")
            report.append(f"  Cache Hits: {info['cache_hits']}")
            report.append(f"  Efficiency: {info['efficiency']:.1f}%")
        report.append("")
        
        # Cache Directory Scan
        report.append("📁 CACHE DIRECTORY SCAN")
        report.append("-" * 40)
        for model_name, info in cache_scan.items():
            report.append(f"{model_name}:")
            report.append(f"  Path: {info['path']}")
            report.append(f"  Files: {info['files']}")
            report.append(f"  Size: {info['size_mb']:.1f} MB")
        report.append("")
        
        # Recent Downloads
        report.append("📥 RECENT DOWNLOADS (Last 10)")
        report.append("-" * 40)
        for event in self.download_log[-10:]:
            status = "CACHE HIT" if event["cache_hit"] else "DOWNLOADED"
            report.append(f"{event['timestamp']} - {event['model_name']} ({event['domain']}): {status} in {event['download_time']:.2f}s")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: Optional[str] = None):
        """Save the report to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_cache_report_{timestamp}.txt"
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📄 Report saved to: {filename}")
        return filename

def main():
    """Main function for standalone usage"""
    tracker = ModelCacheTracker()
    
    # Scan existing cache
    cache_scan = tracker.scan_cache_directory()
    print(f"Found {len(cache_scan)} models in cache")
    
    # Generate and display report
    report = tracker.generate_report()
    print(report)
    
    # Save report
    filename = tracker.save_report()
    print(f"\nReport saved to: {filename}")

if __name__ == "__main__":
    main() 