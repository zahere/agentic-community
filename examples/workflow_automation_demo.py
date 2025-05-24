#!/usr/bin/env python3
"""
Workflow Automation Demo - Agentic Community Edition

This example demonstrates the WorkflowEngine's ability to orchestrate
complex multi-step agent tasks with dependencies, conditions, and branching.
"""

import asyncio
import os
from datetime import datetime, timedelta

from agentic_community.workflow import (
    WorkflowEngine,
    WorkflowTask,
    TaskType,
    TaskStatus
)
from agentic_community.agents import SimpleAgent, ReasoningAgent
from agentic_community.tools import (
    SearchTool,
    EmailTool,
    CalendarTool,
    DatabaseTool
)


async def customer_onboarding_workflow():
    """Demonstrate a customer onboarding workflow."""
    print("=== Customer Onboarding Workflow Demo ===\n")
    
    # Create workflow engine
    engine = WorkflowEngine("CustomerOnboarding")
    
    # Create agents
    data_agent = SimpleAgent(
        name="DataAgent",
        role="Customer data processor",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    data_agent.add_tool(DatabaseTool())
    
    email_agent = SimpleAgent(
        name="EmailAgent",
        role="Email communication specialist",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    email_agent.add_tool(EmailTool())
    
    calendar_agent = SimpleAgent(
        name="CalendarAgent",
        role="Meeting scheduler",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    calendar_agent.add_tool(CalendarTool())
    
    # Register agents
    engine.register_agent("data_agent", data_agent)
    engine.register_agent("email_agent", email_agent)
    engine.register_agent("calendar_agent", calendar_agent)
    
    # Define workflow tasks
    tasks = [
        # 1. Validate customer data
        WorkflowTask(
            id="validate_data",
            name="Validate Customer Data",
            type=TaskType.AGENT,
            config={
                "agent_id": "data_agent",
                "prompt": "Validate customer data: name={customer_name}, email={customer_email}, company={customer_company}"
            }
        ),
        
        # 2. Create customer record (conditional on validation)
        WorkflowTask(
            id="create_record",
            name="Create Customer Record",
            type=TaskType.AGENT,
            config={
                "agent_id": "data_agent",
                "prompt": "Create customer record in database for {customer_name} from {customer_company}"
            },
            dependencies=["validate_data"],
            conditions=[{
                "type": "contains",
                "field": "task_validate_data_output",
                "value": "valid"
            }]
        ),
        
        # 3. Send welcome email
        WorkflowTask(
            id="send_welcome",
            name="Send Welcome Email",
            type=TaskType.AGENT,
            config={
                "agent_id": "email_agent",
                "prompt": "Send welcome email to {customer_email} with onboarding information"
            },
            dependencies=["create_record"]
        ),
        
        # 4. Schedule onboarding meeting
        WorkflowTask(
            id="schedule_meeting",
            name="Schedule Onboarding Meeting",
            type=TaskType.AGENT,
            config={
                "agent_id": "calendar_agent",
                "prompt": "Schedule 30-minute onboarding meeting with {customer_name} next week"
            },
            dependencies=["send_welcome"]
        ),
        
        # 5. Send meeting confirmation
        WorkflowTask(
            id="send_confirmation",
            name="Send Meeting Confirmation",
            type=TaskType.AGENT,
            config={
                "agent_id": "email_agent",
                "prompt": "Send meeting confirmation to {customer_email} with calendar invite details from {task_schedule_meeting_output}"
            },
            dependencies=["schedule_meeting"]
        )
    ]
    
    # Create workflow
    engine.create_workflow("onboarding_v1", tasks)
    
    # Execute workflow with customer data
    context = {
        "customer_name": "John Smith",
        "customer_email": "john.smith@example.com",
        "customer_company": "Acme Corp"
    }
    
    print(f"Starting onboarding for: {context['customer_name']}\n")
    
    results = await engine.execute_workflow("onboarding_v1", context)
    
    # Display results
    print("\nWorkflow Results:")
    print("-" * 50)
    
    for task_id, result in results.items():
        task = next(t for t in tasks if t.id == task_id)
        print(f"\n{task.name}:")
        print(f"  Status: {result.status.value}")
        if result.output:
            print(f"  Output: {result.output[:100]}...")
        if result.error:
            print(f"  Error: {result.error}")
            
    # Get workflow status
    status = engine.get_workflow_status("onboarding_v1")
    print(f"\nWorkflow Status: {status['status']}")
    print(f"Completed Tasks: {status['completed_tasks']}/{status['total_tasks']}")


async def research_workflow():
    """Demonstrate a research and report generation workflow."""
    print("\n\n=== Research Workflow Demo ===\n")
    
    # Create workflow engine
    engine = WorkflowEngine("ResearchWorkflow")
    
    # Create agents
    research_agent = ReasoningAgent(
        name="ResearchAgent",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    research_agent.add_tool(SearchTool())
    
    analyst_agent = ReasoningAgent(
        name="AnalystAgent",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    writer_agent = SimpleAgent(
        name="WriterAgent",
        role="Technical writer",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Register agents
    engine.register_agent("researcher", research_agent)
    engine.register_agent("analyst", analyst_agent)
    engine.register_agent("writer", writer_agent)
    
    # Define parallel research tasks
    research_tasks = [
        WorkflowTask(
            id="research_market",
            name="Research Market Trends",
            type=TaskType.AGENT,
            config={
                "agent_id": "researcher",
                "prompt": "Research current market trends in {industry}"
            }
        ),
        WorkflowTask(
            id="research_competitors",
            name="Research Competitors",
            type=TaskType.AGENT,
            config={
                "agent_id": "researcher",
                "prompt": "Research top competitors in {industry}"
            }
        ),
        WorkflowTask(
            id="research_technology",
            name="Research Technology",
            type=TaskType.AGENT,
            config={
                "agent_id": "researcher",
                "prompt": "Research emerging technologies in {industry}"
            }
        )
    ]
    
    # Define workflow with parallel execution
    tasks = [
        # 1. Parallel research phase
        WorkflowTask(
            id="research_phase",
            name="Conduct Research",
            type=TaskType.PARALLEL,
            config={
                "tasks": research_tasks
            }
        ),
        
        # 2. Analyze findings
        WorkflowTask(
            id="analyze",
            name="Analyze Research Findings",
            type=TaskType.AGENT,
            config={
                "agent_id": "analyst",
                "prompt": "Analyze and synthesize research findings: {task_research_phase_output}"
            },
            dependencies=["research_phase"]
        ),
        
        # 3. Generate report
        WorkflowTask(
            id="write_report",
            name="Write Research Report",
            type=TaskType.AGENT,
            config={
                "agent_id": "writer",
                "prompt": "Write comprehensive research report based on analysis: {task_analyze_output}"
            },
            dependencies=["analyze"]
        )
    ]
    
    # Create and execute workflow
    engine.create_workflow("research_v1", tasks)
    
    context = {
        "industry": "AI and Machine Learning"
    }
    
    print(f"Researching: {context['industry']}\n")
    
    results = await engine.execute_workflow("research_v1", context)
    
    # Display final report
    report_result = results.get("write_report")
    if report_result and report_result.status == TaskStatus.COMPLETED:
        print("\nGenerated Research Report:")
        print("=" * 50)
        print(report_result.output)


async def conditional_workflow():
    """Demonstrate conditional branching in workflows."""
    print("\n\n=== Conditional Workflow Demo ===\n")
    
    engine = WorkflowEngine("ConditionalWorkflow")
    
    # Create decision-making agent
    decision_agent = ReasoningAgent(
        name="DecisionAgent",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    action_agent = SimpleAgent(
        name="ActionAgent",
        role="Action executor",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    engine.register_agent("decision", decision_agent)
    engine.register_agent("action", action_agent)
    
    # Define workflow with conditional branches
    tasks = [
        # 1. Analyze situation
        WorkflowTask(
            id="analyze_situation",
            name="Analyze Situation",
            type=TaskType.AGENT,
            config={
                "agent_id": "decision",
                "prompt": "Analyze risk level for investment amount: ${amount}"
            }
        ),
        
        # 2. High risk action (conditional)
        WorkflowTask(
            id="high_risk_action",
            name="High Risk Mitigation",
            type=TaskType.AGENT,
            config={
                "agent_id": "action",
                "prompt": "Implement high risk mitigation strategy"
            },
            dependencies=["analyze_situation"],
            conditions=[{
                "type": "contains",
                "field": "task_analyze_situation_output",
                "value": "high risk"
            }]
        ),
        
        # 3. Low risk action (conditional)
        WorkflowTask(
            id="low_risk_action",
            name="Standard Processing",
            type=TaskType.AGENT,
            config={
                "agent_id": "action",
                "prompt": "Proceed with standard investment processing"
            },
            dependencies=["analyze_situation"],
            conditions=[{
                "type": "contains",
                "field": "task_analyze_situation_output",
                "value": "low risk"
            }]
        ),
        
        # 4. Final step (runs regardless)
        WorkflowTask(
            id="finalize",
            name="Finalize Process",
            type=TaskType.AGENT,
            config={
                "agent_id": "action",
                "prompt": "Finalize investment process and generate confirmation"
            },
            dependencies=["high_risk_action", "low_risk_action"]
        )
    ]
    
    # Test with different amounts
    amounts = [1000, 50000, 1000000]
    
    for amount in amounts:
        print(f"\nTesting with amount: ${amount:,}")
        
        engine.create_workflow(f"invest_{amount}", tasks)
        results = await engine.execute_workflow(f"invest_{amount}", {"amount": amount})
        
        # Show which branch was taken
        if results["high_risk_action"].status == TaskStatus.COMPLETED:
            print("  → High risk path taken")
        elif results["low_risk_action"].status == TaskStatus.COMPLETED:
            print("  → Low risk path taken")
        else:
            print("  → No action taken")


async def main():
    """Run all workflow demos."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Run customer onboarding workflow
        await customer_onboarding_workflow()
        
        # Run research workflow
        await research_workflow()
        
        # Run conditional workflow
        await conditional_workflow()
        
        print("\n\n✅ All workflow demonstrations completed!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
