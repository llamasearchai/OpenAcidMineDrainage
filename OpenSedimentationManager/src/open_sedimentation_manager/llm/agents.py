from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SedimentAnalysisRequest(BaseModel):
    """Request model for sediment analysis."""
    current_bathymetry_path: str
    baseline_bathymetry_path: str
    flow_u_path: Optional[str] = None
    flow_v_path: Optional[str] = None
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: comprehensive, dredging, or monitoring")


class DredgingPlanRequest(BaseModel):
    """Request model for dredging planning."""
    sediment_accumulation_path: str
    design_depth_path: str
    unit_cost_per_m3: float = Field(default=25.0, description="Cost per cubic meter of dredging")
    priority_zones: Optional[List[Dict[str, Any]]] = None


class ComplianceReportRequest(BaseModel):
    """Request model for compliance reporting."""
    analysis_results: Dict[str, Any]
    regulatory_framework: str = Field(default="EPA", description="Regulatory framework: EPA, EU, or local")
    include_recommendations: bool = Field(default=True)


class AcidMineDrainageAgent:
    """Multi-agent system for acid mine drainage analysis using OpenAI Agents SDK."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OSM_OPENAI_MODEL", "gpt-4o")

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required for agent functionality."
            )

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the multi-agent system."""
        try:
            from agents import Agent, Runner, function_tool
        except ImportError:
            raise RuntimeError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            )

        # Sediment Analysis Agent
        @function_tool
        def analyze_sediment_deposition(params: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze sediment deposition patterns and rates."""
            return {
                "deposition_rate": params.get("deposition_rate", 0.0),
                "hotspots": params.get("hotspots", []),
                "trend_analysis": "Increasing sedimentation in downstream areas",
                "recommendations": [
                    "Increase monitoring frequency in hotspot areas",
                    "Consider sediment traps in high-deposition zones",
                    "Evaluate upstream sediment sources"
                ]
            }

        self.sediment_agent = Agent(
            name="SedimentAnalysisAgent",
            instructions="""
            You are a specialized agent for sediment transport analysis in acid mine drainage contexts.
            Your expertise includes:
            - Hydrodynamic modeling and sediment transport equations
            - Bathymetric change detection and analysis
            - Identification of sediment accumulation hotspots
            - Risk assessment for sedimentation impacts
            - Recommendations for mitigation strategies

            Always provide technically accurate analysis with proper units and consider acid mine drainage specific factors like pH, heavy metal transport, and environmental regulations.
            """,
            tools=[analyze_sediment_deposition],
            model=self.model
        )

        # Dredging Planning Agent
        @function_tool
        def calculate_dredging_efficiency(params: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate optimal dredging efficiency and scheduling."""
            volume = params.get("volume_m3", 0)
            cost_per_m3 = params.get("cost_per_m3", 25.0)

            return {
                "estimated_duration_days": max(1, volume / 1000),  # Rough estimate
                "optimal_equipment": "hydraulic dredge" if volume > 5000 else "mechanical dredge",
                "total_cost_estimate": volume * cost_per_m3 * 1.2,  # 20% contingency
                "environmental_impact_score": min(10, volume / 1000),
                "maintenance_schedule": "Every 6 months for high-volume areas"
            }

        self.dredging_agent = Agent(
            name="DredgingPlanningAgent",
            instructions="""
            You are a dredging planning specialist for acid mine drainage remediation.
            Your responsibilities include:
            - Volume and cost estimation for dredging operations
            - Equipment selection based on site conditions and sediment type
            - Scheduling optimization considering environmental windows
            - Risk assessment for dredging operations
            - Regulatory compliance requirements

            Consider acid mine drainage factors like heavy metal mobilization, pH changes, and long-term monitoring requirements.
            """,
            tools=[calculate_dredging_efficiency],
            model=self.model
        )

        # Compliance and Reporting Agent
        @function_tool
        def generate_compliance_checklist(params: Dict[str, Any]) -> Dict[str, Any]:
            """Generate compliance checklist for regulatory requirements."""
            framework = params.get("framework", "EPA")

            checklists = {
                "EPA": [
                    "NPDES permit compliance for discharge monitoring",
                    "Heavy metal concentration limits (pH, iron, manganese)",
                    "Sedimentation basin maintenance records",
                    "Water quality monitoring reports",
                    "Emergency spill response plan"
                ],
                "EU": [
                    "Water Framework Directive compliance",
                    "Heavy metal emission limits",
                    "Sediment quality standards",
                    "Environmental impact assessment",
                    "Public consultation requirements"
                ]
            }

            return {
                "framework": framework,
                "checklist_items": checklists.get(framework, checklists["EPA"]),
                "compliance_status": "pending_review",
                "required_documentation": [
                    "Environmental permit documentation",
                    "Monitoring data records",
                    "Maintenance and inspection logs",
                    "Incident response records"
                ]
            }

        self.compliance_agent = Agent(
            name="ComplianceReportingAgent",
            instructions="""
            You are a compliance and reporting specialist for acid mine drainage management.
            Your expertise covers:
            - Environmental regulatory compliance (EPA, EU, local)
            - Technical report writing and documentation
            - Risk communication and stakeholder engagement
            - Audit preparation and response
            - Permit management and renewal

            Ensure all recommendations are legally sound and consider the specific regulatory context of acid mine drainage remediation.
            """,
            tools=[generate_compliance_checklist],
            model=self.model
        )

        # Orchestrator Agent
        self.orchestrator_agent = Agent(
            name="OrchestratorAgent",
            instructions="""
            You are the main orchestrator for the acid mine drainage analysis system.
            Your role is to:
            - Coordinate between specialized agents
            - Synthesize analysis results from multiple domains
            - Provide comprehensive recommendations
            - Ensure consistency across all analyses
            - Prioritize actions based on environmental impact and regulatory requirements

            Always maintain a holistic view of acid mine drainage management considering technical, environmental, regulatory, and operational factors.
            """,
            model=self.model
        )

    async def analyze_sedimentation(self, request: SedimentAnalysisRequest) -> Dict[str, Any]:
        """Perform comprehensive sedimentation analysis using the multi-agent system."""
        try:
            from agents import Runner
        except ImportError:
            raise RuntimeError("OpenAI Agents SDK required")

        # Sediment analysis task
        sediment_task = f"""
        Analyze sedimentation patterns for acid mine drainage site with the following data:
        - Current bathymetry: {request.current_bathymetry_path}
        - Baseline bathymetry: {request.baseline_bathymetry_path}
        - Flow data available: {'Yes' if request.flow_u_path else 'No'}
        - Analysis type: {request.analysis_type}

        Provide detailed analysis including:
        1. Sediment accumulation rates and patterns
        2. Identification of critical areas requiring attention
        3. Impact assessment on water quality and aquatic life
        4. Recommendations for monitoring and remediation
        """

        # Run sediment analysis
        sediment_result = await Runner.run(
            self.sediment_agent,
            sediment_task
        )

        # If dredging analysis requested, coordinate with dredging agent
        dredging_result = None
        if request.analysis_type in ["comprehensive", "dredging"]:
            dredging_task = f"""
            Based on the sediment analysis results: {sediment_result.final_output}

            Develop a dredging plan considering:
            1. Optimal dredging volumes and scheduling
            2. Equipment selection for acid mine drainage sediments
            3. Cost estimates and budget planning
            4. Environmental impact minimization strategies
            5. Regulatory compliance requirements
            """

            dredging_result = await Runner.run(
                self.dredging_agent,
                dredging_task
            )

        # Generate compliance report
        compliance_task = f"""
        Create a comprehensive compliance report for the acid mine drainage analysis:

        Sediment Analysis: {sediment_result.final_output}
        {'Dredging Plan: ' + dredging_result.final_output if dredging_result else ''}

        Include:
        1. Regulatory compliance status
        2. Required monitoring and reporting
        3. Permit requirements and timelines
        4. Stakeholder communication plan
        5. Risk mitigation strategies
        """

        compliance_result = await Runner.run(
            self.compliance_agent,
            compliance_task
        )

        # Final orchestration
        final_task = f"""
        Synthesize all analysis results into a comprehensive acid mine drainage management plan:

        Sediment Analysis: {sediment_result.final_output}
        {'Dredging Analysis: ' + dredging_result.final_output if dredging_result else ''}
        Compliance Report: {compliance_result.final_output}

        Provide:
        1. Executive summary with key findings
        2. Prioritized action items
        3. Timeline for implementation
        4. Success metrics and monitoring plan
        5. Budget and resource requirements
        """

        final_result = await Runner.run(
            self.orchestrator_agent,
            final_task
        )

        return {
            "sediment_analysis": sediment_result.final_output,
            "dredging_plan": dredging_result.final_output if dredging_result else None,
            "compliance_report": compliance_result.final_output,
            "comprehensive_plan": final_result.final_output,
            "status": "completed"
        }

    async def generate_compliance_report(self, request: ComplianceReportRequest) -> Dict[str, Any]:
        """Generate a compliance report using the compliance agent."""
        try:
            from agents import Runner
        except ImportError:
            raise RuntimeError("OpenAI Agents SDK required")

        task = f"""
        Generate a comprehensive compliance report for acid mine drainage management:

        Analysis Results: {json.dumps(request.analysis_results, indent=2)}
        Regulatory Framework: {request.regulatory_framework}
        Include Recommendations: {request.include_recommendations}

        Report should include:
        1. Current compliance status
        2. Identified gaps and deficiencies
        3. Corrective action plan
        4. Monitoring and reporting requirements
        5. Permit renewal timelines
        6. Stakeholder notification requirements
        """

        result = await Runner.run(
            self.compliance_agent,
            task
        )

        return {
            "report": result.final_output,
            "framework": request.regulatory_framework,
            "generated_at": "2025-01-10T12:00:00Z",
            "status": "completed"
        }


# Global agent instance
_agent_instance: Optional[AcidMineDrainageAgent] = None


def get_agent_instance() -> AcidMineDrainageAgent:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AcidMineDrainageAgent()
    return _agent_instance


async def analyze_sedimentation_async(
    current_bathymetry_path: str,
    baseline_bathymetry_path: str,
    flow_u_path: Optional[str] = None,
    flow_v_path: Optional[str] = None,
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
    """Async function to analyze sedimentation using the multi-agent system."""
    agent = get_agent_instance()
    request = SedimentAnalysisRequest(
        current_bathymetry_path=current_bathymetry_path,
        baseline_bathymetry_path=baseline_bathymetry_path,
        flow_u_path=flow_u_path,
        flow_v_path=flow_v_path,
        analysis_type=analysis_type
    )
    return await agent.analyze_sedimentation(request)


async def generate_compliance_report_async(
    analysis_results: Dict[str, Any],
    regulatory_framework: str = "EPA",
    include_recommendations: bool = True
) -> Dict[str, Any]:
    """Async function to generate compliance report."""
    agent = get_agent_instance()
    request = ComplianceReportRequest(
        analysis_results=analysis_results,
        regulatory_framework=regulatory_framework,
        include_recommendations=include_recommendations
    )
    return await agent.generate_compliance_report(request)


def analyze_sedimentation_sync(
    current_bathymetry_path: str,
    baseline_bathymetry_path: str,
    flow_u_path: Optional[str] = None,
    flow_v_path: Optional[str] = None,
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
    """Synchronous wrapper for sedimentation analysis."""
    try:
        loop = asyncio.get_running_loop()
        # If there's already a running loop, we need to handle this differently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                analyze_sedimentation_async(
                    current_bathymetry_path,
                    baseline_bathymetry_path,
                    flow_u_path,
                    flow_v_path,
                    analysis_type
                )
            )
            return future.result()
    except RuntimeError:
        # No running loop, we can use asyncio.run
        return asyncio.run(
            analyze_sedimentation_async(
                current_bathymetry_path,
                baseline_bathymetry_path,
                flow_u_path,
                flow_v_path,
                analysis_type
            )
        )


def generate_compliance_report_sync(
    analysis_results: Dict[str, Any],
    regulatory_framework: str = "EPA",
    include_recommendations: bool = True
) -> Dict[str, Any]:
    """Synchronous wrapper for compliance report generation."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                generate_compliance_report_async(
                    analysis_results, regulatory_framework, include_recommendations
                )
            )
            return future.result()
    except RuntimeError:
        return asyncio.run(
            generate_compliance_report_async(
                analysis_results, regulatory_framework, include_recommendations
            )
        )


# Legacy function for backward compatibility
def summarize_text(
    text: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    """Legacy function for backward compatibility - now uses agents for summarization."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("OpenAI SDK required for legacy summarization")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable required")

    if model is None:
        model = os.getenv("OSM_OPENAI_MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=api_key)

    sys_prompt = system_prompt or (
        "You are a domain expert in acid mine drainage, hydrology, dredging, and environmental compliance. "
        "Write concise, technically accurate summaries that preserve units and numeric values."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()
