"""
Collaborative Agent for Community Edition

This agent can work with other agents to solve problems collaboratively.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from agentic_community.core.base import BaseAgent
from agentic_community.core.exceptions import AgentError
from agentic_community.core.llm_providers import create_llm_client


@dataclass
class CollaborationResult:
    """Result of a collaborative effort"""
    task: str
    agents: List[str]
    individual_responses: Dict[str, str]
    synthesized_response: str
    timestamp: datetime
    consensus_score: float


class CollaborativeAgent(BaseAgent):
    """
    Agent that can collaborate with other agents to solve problems.
    
    Features:
    - Multi-agent collaboration
    - Response synthesis
    - Consensus building
    - Conflict resolution
    """
    
    def __init__(self, name: str, llm_provider: str = "openai", **kwargs):
        super().__init__(name)
        self.llm_client = create_llm_client(llm_provider, **kwargs)
        self.collaboration_history = []
        
    async def collaborate(
        self,
        other_agents: List['CollaborativeAgent'],
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CollaborationResult:
        """
        Collaborate with other agents on a task.
        
        Args:
            other_agents: List of agents to collaborate with
            task: The task to collaborate on
            context: Optional context for the task
            
        Returns:
            CollaborationResult with synthesized response
        """
        all_agents = [self] + other_agents
        agent_names = [agent.name for agent in all_agents]
        
        # Gather individual responses
        individual_responses = {}
        tasks = []
        
        for agent in all_agents:
            tasks.append(agent._generate_response(task, context))
        
        responses = await asyncio.gather(*tasks)
        
        for agent, response in zip(all_agents, responses):
            individual_responses[agent.name] = response
        
        # Synthesize responses
        synthesized = await self._synthesize_responses(
            task,
            individual_responses,
            context
        )
        
        # Calculate consensus score
        consensus_score = await self._calculate_consensus(
            individual_responses,
            synthesized
        )
        
        result = CollaborationResult(
            task=task,
            agents=agent_names,
            individual_responses=individual_responses,
            synthesized_response=synthesized,
            timestamp=datetime.now(),
            consensus_score=consensus_score
        )
        
        self.collaboration_history.append(result)
        return result
    
    async def _generate_response(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate individual response to a task.
        """
        messages = [
            {"role": "system", "content": f"You are {self.name}, a collaborative agent."},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"Context: {json.dumps(context)}"
            })
        
        response = await self.llm_client.complete(messages)
        return response.content
    
    async def _synthesize_responses(
        self,
        task: str,
        responses: Dict[str, str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesize multiple agent responses into a unified solution.
        """
        synthesis_prompt = f"""
        Task: {task}
        
        Multiple agents have provided the following perspectives:
        
        {self._format_responses(responses)}
        
        Please synthesize these perspectives into a comprehensive solution that:
        1. Incorporates the best insights from each agent
        2. Resolves any contradictions or conflicts
        3. Provides a clear, actionable response
        """
        
        messages = [
            {"role": "system", "content": "You are a synthesis expert."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        response = await self.llm_client.complete(messages)
        return response.content
    
    def _format_responses(self, responses: Dict[str, str]) -> str:
        """Format responses for synthesis prompt."""
        formatted = []
        for agent_name, response in responses.items():
            formatted.append(f"**{agent_name}**:\n{response}\n")
        return "\n".join(formatted)
    
    async def _calculate_consensus(
        self,
        individual_responses: Dict[str, str],
        synthesized_response: str
    ) -> float:
        """
        Calculate consensus score between individual and synthesized responses.
        """
        # Simple implementation - could be enhanced with semantic similarity
        # For now, check overlap of key terms
        
        synthesized_words = set(synthesized_response.lower().split())
        total_overlap = 0
        
        for response in individual_responses.values():
            response_words = set(response.lower().split())
            overlap = len(synthesized_words & response_words) / len(response_words)
            total_overlap += overlap
        
        consensus_score = total_overlap / len(individual_responses)
        return min(consensus_score, 1.0)
    
    async def negotiate(
        self,
        other_agent: 'CollaborativeAgent',
        negotiation_task: str,
        max_rounds: int = 3
    ) -> Dict[str, Any]:
        """
        Negotiate with another agent to reach agreement.
        
        Args:
            other_agent: Agent to negotiate with
            negotiation_task: The negotiation topic
            max_rounds: Maximum negotiation rounds
            
        Returns:
            Negotiation result with final agreement
        """
        rounds = []
        current_proposal = None
        
        for round_num in range(max_rounds):
            # Get proposal from self
            self_proposal = await self._make_proposal(
                negotiation_task,
                current_proposal,
                f"{other_agent.name}'s perspective"
            )
            
            # Get counter-proposal from other agent
            other_proposal = await other_agent._make_proposal(
                negotiation_task,
                self_proposal,
                f"{self.name}'s perspective"
            )
            
            rounds.append({
                "round": round_num + 1,
                f"{self.name}_proposal": self_proposal,
                f"{other_agent.name}_proposal": other_proposal
            })
            
            # Check for agreement
            if await self._check_agreement(self_proposal, other_proposal):
                return {
                    "agreed": True,
                    "final_agreement": other_proposal,
                    "rounds": rounds,
                    "total_rounds": round_num + 1
                }
            
            current_proposal = other_proposal
        
        # No agreement reached
        return {
            "agreed": False,
            "rounds": rounds,
            "total_rounds": max_rounds,
            "final_proposals": {
                self.name: rounds[-1][f"{self.name}_proposal"],
                other_agent.name: rounds[-1][f"{other_agent.name}_proposal"]
            }
        }
    
    async def _make_proposal(
        self,
        task: str,
        previous_proposal: Optional[str],
        other_perspective: str
    ) -> str:
        """
        Make a negotiation proposal.
        """
        prompt = f"""
        Negotiation task: {task}
        
        You need to make a proposal considering {other_perspective}.
        """
        
        if previous_proposal:
            prompt += f"\n\nPrevious proposal:\n{previous_proposal}\n\nPlease refine or counter this proposal."
        else:
            prompt += "\n\nMake an initial proposal."
        
        messages = [
            {"role": "system", "content": f"You are {self.name}, a negotiating agent."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm_client.complete(messages)
        return response.content
    
    async def _check_agreement(
        self,
        proposal1: str,
        proposal2: str
    ) -> bool:
        """
        Check if two proposals represent agreement.
        """
        messages = [
            {"role": "system", "content": "You are an agreement evaluator."},
            {"role": "user", "content": f"""
            Do these two proposals represent substantial agreement?
            
            Proposal 1: {proposal1}
            
            Proposal 2: {proposal2}
            
            Respond with only 'YES' or 'NO'.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        return "YES" in response.content.upper()
    
    async def brainstorm(
        self,
        other_agents: List['CollaborativeAgent'],
        topic: str,
        num_ideas: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Brainstorm ideas collaboratively.
        
        Args:
            other_agents: Agents to brainstorm with
            topic: Brainstorming topic
            num_ideas: Number of ideas to generate per agent
            
        Returns:
            List of ideas with ratings
        """
        all_agents = [self] + other_agents
        all_ideas = []
        
        # Generate ideas from each agent
        for agent in all_agents:
            ideas = await agent._generate_ideas(topic, num_ideas)
            for idea in ideas:
                all_ideas.append({
                    "idea": idea,
                    "author": agent.name,
                    "ratings": {}
                })
        
        # Have each agent rate all ideas
        for agent in all_agents:
            for idea_dict in all_ideas:
                if idea_dict["author"] != agent.name:  # Don't rate own ideas
                    rating = await agent._rate_idea(
                        idea_dict["idea"],
                        topic
                    )
                    idea_dict["ratings"][agent.name] = rating
        
        # Calculate average ratings
        for idea_dict in all_ideas:
            if idea_dict["ratings"]:
                avg_rating = sum(idea_dict["ratings"].values()) / len(idea_dict["ratings"])
                idea_dict["average_rating"] = avg_rating
            else:
                idea_dict["average_rating"] = 0
        
        # Sort by rating
        all_ideas.sort(key=lambda x: x["average_rating"], reverse=True)
        
        return all_ideas
    
    async def _generate_ideas(
        self,
        topic: str,
        num_ideas: int
    ) -> List[str]:
        """
        Generate brainstorming ideas.
        """
        messages = [
            {"role": "system", "content": f"You are {self.name}, a creative agent."},
            {"role": "user", "content": f"""
            Generate {num_ideas} creative ideas for: {topic}
            
            Format: One idea per line, no numbering.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        ideas = [line.strip() for line in response.content.split('\n') if line.strip()]
        return ideas[:num_ideas]
    
    async def _rate_idea(
        self,
        idea: str,
        topic: str
    ) -> float:
        """
        Rate an idea on a scale of 0-1.
        """
        messages = [
            {"role": "system", "content": "You are an idea evaluator."},
            {"role": "user", "content": f"""
            Rate this idea for '{topic}' on a scale of 0-10:
            
            {idea}
            
            Consider: originality, feasibility, and impact.
            Respond with only a number.
            """}
        ]
        
        response = await self.llm_client.complete(messages)
        try:
            rating = float(response.content.strip()) / 10
            return min(max(rating, 0), 1)  # Ensure 0-1 range
        except:
            return 0.5  # Default rating
    
    async def process(self, input_data: str) -> str:
        """
        Process input using the collaborative agent.
        """
        # For single agent mode, just generate a response
        return await self._generate_response(input_data)


# Example usage
if __name__ == "__main__":
    async def demo_collaboration():
        # Create collaborative agents
        alice = CollaborativeAgent("Alice", llm_provider="openai")
        bob = CollaborativeAgent("Bob", llm_provider="openai")
        charlie = CollaborativeAgent("Charlie", llm_provider="openai")
        
        # Test collaboration
        print("=== Collaboration Demo ===")
        result = await alice.collaborate(
            [bob, charlie],
            "How can we improve software development productivity?"
        )
        
        print(f"\nSynthesized Response:\n{result.synthesized_response}")
        print(f"\nConsensus Score: {result.consensus_score:.2f}")
        
        # Test negotiation
        print("\n=== Negotiation Demo ===")
        negotiation = await alice.negotiate(
            bob,
            "Decide on the best programming language for a new web application"
        )
        
        if negotiation["agreed"]:
            print(f"Agreement reached in {negotiation['total_rounds']} rounds!")
            print(f"Final agreement: {negotiation['final_agreement']}")
        else:
            print("No agreement reached.")
        
        # Test brainstorming
        print("\n=== Brainstorming Demo ===")
        ideas = await alice.brainstorm(
            [bob, charlie],
            "Innovative features for an AI assistant",
            num_ideas=3
        )
        
        print("\nTop Ideas:")
        for i, idea in enumerate(ideas[:5]):
            print(f"{i+1}. {idea['idea']} (by {idea['author']}, rating: {idea['average_rating']:.2f})")
    
    # Run the demo
    asyncio.run(demo_collaboration())
