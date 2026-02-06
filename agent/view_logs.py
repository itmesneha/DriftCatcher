"""
View agent logs in human-readable format
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp to readable string"""
    dt = datetime.fromisoformat(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def view_reasoning_decisions(limit: int = 10):
    """View recent reasoning engine decisions"""
    log_file = Path("agent/logs/reasoning_decisions.jsonl")
    
    if not log_file.exists():
        print("No reasoning decisions logged yet.")
        return
    
    print("\n" + "="*80)
    print("REASONING ENGINE DECISION LOG")
    print("="*80)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines[-limit:]:
        entry = json.loads(line)
        
        print(f"\n⏰ Time: {format_timestamp(entry['timestamp'])}")
        print(f"📊 Drift PSI: {entry['drift_results'].get('overall_psi', 'N/A'):.4f}")
        print(f"🔢 Drifted Features: {entry['drift_results'].get('n_drifted_features', 'N/A')}")
        print(f"\n💡 Decision:")
        print(f"   Action: {entry['decision']['action']}")
        print(f"   Confidence: {entry['decision']['confidence']:.2f}")
        print(f"   Type: {entry['decision'].get('reasoning_type', 'rule-based')}")
        print(f"\n📝 Reasoning:")
        print(f"   {entry['decision']['reasoning']}")
        print(f"\n🎯 Thresholds at decision time:")
        print(f"   PSI Low: {entry['thresholds']['psi_low']:.3f}")
        print(f"   PSI High: {entry['thresholds']['psi_high']:.3f}")
        print("-" * 80)


def view_planning_plans(limit: int = 5):
    """View recent plans created by planning agent"""
    log_file = Path("agent/logs/planning_plans.jsonl")
    
    if not log_file.exists():
        print("No planning logs yet.")
        return
    
    print("\n" + "="*80)
    print("PLANNING AGENT PLANS")
    print("="*80)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines[-limit:]:
        entry = json.loads(line)
        
        print(f"\n⏰ Time: {format_timestamp(entry['timestamp'])}")
        print(f"🎯 Goal: {entry['goal']}")
        print(f"\n📋 Plan ({len(entry['plan'])} steps):")
        
        for step in entry['plan']:
            deps = f" (depends on: {step['dependencies']})" if step['dependencies'] else ""
            print(f"   {step['step_id']}. {step['description']}{deps}")
            print(f"      Tool: {step['tool_name']}")
        
        print("-" * 80)


def view_planning_executions(limit: int = 3):
    """View recent plan executions"""
    log_file = Path("agent/logs/planning_executions.jsonl")
    
    if not log_file.exists():
        print("No execution logs yet.")
        return
    
    print("\n" + "="*80)
    print("PLAN EXECUTION RESULTS")
    print("="*80)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines[-limit:]:
        entry = json.loads(line)
        
        print(f"\n🎯 Plan ID: {entry['plan_id']}")
        print(f"📊 Summary:")
        print(f"   Total Steps: {entry['total_steps']}")
        print(f"   ✅ Completed: {entry['completed_steps']}")
        print(f"   ❌ Failed: {entry['failed_steps']}")
        
        print(f"\n📝 Step Results:")
        for result in entry['step_results']:
            status_emoji = {
                'completed': '✅',
                'failed': '❌',
                'blocked': '🚫',
                'skipped': '⏭️'
            }.get(result['status'], '❓')
            
            print(f"   {status_emoji} Step {result['step_id']}: {result['tool']}")
            
            if result.get('error'):
                print(f"      Error: {result['error']}")
            elif result.get('result', {}).get('skipped'):
                print(f"      Skipped: {result['result'].get('reason', 'N/A')}")
        
        print("-" * 80)


def main():
    """View all logs"""
    print("\n" + "🤖 AGENTIC AI SYSTEM LOGS " + "🤖")
    
    view_reasoning_decisions(limit=10)
    view_planning_plans(limit=5)
    view_planning_executions(limit=3)
    
    print("\n" + "="*80)
    print("📁 Raw log files available at:")
    print("   - agent/logs/reasoning_decisions.jsonl")
    print("   - agent/logs/planning_plans.jsonl")
    print("   - agent/logs/planning_executions.jsonl")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View agent logs")
    parser.add_argument('--type', choices=['reasoning', 'plans', 'executions', 'all'], 
                       default='all', help='Type of logs to view')
    parser.add_argument('--limit', type=int, default=10, 
                       help='Number of recent entries to show')
    
    args = parser.parse_args()
    
    if args.type == 'reasoning' or args.type == 'all':
        view_reasoning_decisions(limit=args.limit)
    
    if args.type == 'plans' or args.type == 'all':
        view_planning_plans(limit=args.limit)
    
    if args.type == 'executions' or args.type == 'all':
        view_planning_executions(limit=args.limit)
    
    if args.type == 'all':
        print("\n" + "="*80)
        print("📁 Raw log files available at:")
        print("   - agent/logs/reasoning_decisions.jsonl")
        print("   - agent/logs/planning_plans.jsonl")
        print("   - agent/logs/planning_executions.jsonl")
        print("="*80 + "\n")
