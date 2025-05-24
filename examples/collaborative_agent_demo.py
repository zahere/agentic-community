#!/usr/bin/env python3
"""
Collaborative Agent Demo - Agentic Community Edition

This example demonstrates the CollaborativeAgent's ability to work with multiple
agents to solve complex problems through negotiation, brainstorming, and consensus building.
"""

import asyncio
import os
from agentic_community.agents.collaborative_agent import CollaborativeAgent
from agentic_community.agents.simple_agent import SimpleAgent


async def brainstorming_demo():
    """Demonstrate collaborative brainstorming between agents."""
    print("=== Collaborative Brainstorming Demo ===\n")
    
    # Create collaborative agent
    collab_agent = CollaborativeAgent(
        name="BrainstormFacilitator",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Create participant agents
    creative_agent = SimpleAgent(
        name="CreativeExpert",
        role="Creative thinking specialist",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    analytical_agent = SimpleAgent(
        name="AnalyticalExpert", 
        role="Analytical thinking specialist",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Add participants
    collab_agent.add_participant(creative_agent, "creative_perspective")
    collab_agent.add_participant(analytical_agent, "analytical_perspective")
    
    # Brainstorm ideas
    topic = "How can we make remote work more engaging for teams?"
    ideas = await collab_agent.brainstorm(
        topic=topic,
        rounds=2,
        ideas_per_round=3
    )
    
    print(f"Topic: {topic}\n")
    print("Generated Ideas:")
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea}")


async def negotiation_demo():
    """Demonstrate negotiation between agents."""
    print("\n\n=== Negotiation Demo ===\n")
    
    # Create collaborative agent
    negotiator = CollaborativeAgent(
        name="NegotiationMediator",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Create negotiating parties
    buyer_agent = SimpleAgent(
        name="BuyerAgent",
        role="Buyer representative seeking best price",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    seller_agent = SimpleAgent(
        name="SellerAgent",
        role="Seller representative maximizing profit",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Add participants
    negotiator.add_participant(buyer_agent, "buyer_position")
    negotiator.add_participant(seller_agent, "seller_position")
    
    # Negotiate
    issue = "Price for a bulk order of 1000 units of product X"
    initial_positions = {
        "buyer_position": "$50 per unit",
        "seller_position": "$80 per unit"
    }
    
    result = await negotiator.negotiate(
        issue=issue,
        initial_positions=initial_positions,
        max_rounds=3
    )
    
    print(f"Negotiation Issue: {issue}")
    print(f"Initial Positions: {initial_positions}")
    print(f"\nFinal Agreement: {result.get('agreement', 'No agreement reached')}")
    print(f"Consensus Reached: {result.get('consensus', False)}")


async def consensus_building_demo():
    """Demonstrate consensus building among multiple agents."""
    print("\n\n=== Consensus Building Demo ===\n")
    
    # Create collaborative agent
    consensus_builder = CollaborativeAgent(
        name="ConsensusCoordinator",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # Create team members with different perspectives
    tech_lead = SimpleAgent(
        name="TechLead",
        role="Technical architecture expert",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    product_manager = SimpleAgent(
        name="ProductManager",
        role="Product strategy and user experience expert",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    designer = SimpleAgent(
        name="Designer",
        role="UI/UX design specialist",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Add participants
    consensus_builder.add_participant(tech_lead, "technical_view")
    consensus_builder.add_participant(product_manager, "product_view")
    consensus_builder.add_participant(designer, "design_view")
    
    # Build consensus
    question = "Should we build a native mobile app or a progressive web app for our new product?"
    
    consensus = await consensus_builder.build_consensus(
        question=question,
        require_unanimous=False,
        max_rounds=3
    )
    
    print(f"Question: {question}\n")
    print("Consensus Building Results:")
    print(f"Final Decision: {consensus.get('decision', 'No consensus reached')}")
    print(f"Agreement Level: {consensus.get('agreement_level', 0)*100:.1f}%")
    print(f"\nRationale: {consensus.get('rationale', 'N/A')}")


async def main():
    """Run all collaborative agent demos."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Run brainstorming demo
        await brainstorming_demo()
        
        # Run negotiation demo
        await negotiation_demo()
        
        # Run consensus building demo
        await consensus_building_demo()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())