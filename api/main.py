"""
FastAPI server for Neuro-Symbolic Law Prover.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio
import uuid
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuro_symbolic_law import LegalProver, ContractParser
from neuro_symbolic_law.regulations import GDPR, AIAct, CCPA
from neuro_symbolic_law.core.compliance_result import ComplianceStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Neuro-Symbolic Law Prover API",
    description="AI-powered legal compliance verification API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
parser = ContractParser()
prover = LegalProver()
regulations = {
    "gdpr": GDPR(),
    "ai_act": AIAct(),
    "ccpa": CCPA()
}

# In-memory storage for demo (use Redis/database in production)
analysis_results = {}

# Request/Response models
class ContractAnalysisRequest(BaseModel):
    contract_text: str
    contract_id: Optional[str] = None
    regulations: List[str] = ["gdpr"]
    focus_areas: Optional[List[str]] = None

class ComplianceResultResponse(BaseModel):
    requirement_id: str
    requirement_description: str
    status: str
    confidence: float
    issue: Optional[str] = None
    suggestion: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    contract_id: str
    status: str
    compliance_results: Dict[str, List[ComplianceResultResponse]]
    overall_compliance_rate: float
    timestamp: str

class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: Optional[str] = None

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "neuro-symbolic-law-prover"}

# Contract analysis endpoints
@app.post("/analyze", response_model=Dict[str, str])
async def analyze_contract(
    request: ContractAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start contract analysis (async processing).
    
    Returns analysis ID for status tracking.
    """
    analysis_id = str(uuid.uuid4())
    
    # Store initial status
    analysis_results[analysis_id] = AnalysisStatus(
        analysis_id=analysis_id,
        status="pending",
        progress=0,
        message="Analysis queued"
    )
    
    # Start background analysis
    background_tasks.add_task(
        perform_analysis,
        analysis_id,
        request
    )
    
    return {
        "analysis_id": analysis_id,
        "status": "accepted",
        "message": "Analysis started"
    }

@app.get("/analyze/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get analysis status."""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/analyze/{analysis_id}/results")
async def get_analysis_results(analysis_id: str):
    """Get completed analysis results."""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status_obj = analysis_results[analysis_id]
    
    if isinstance(status_obj, AnalysisStatus):
        if status_obj.status != "completed":
            raise HTTPException(
                status_code=202, 
                detail=f"Analysis not ready. Status: {status_obj.status}"
            )
        raise HTTPException(status_code=500, detail="Results not available")
    
    return status_obj  # This should be AnalysisResponse

# Synchronous analysis endpoint
@app.post("/analyze/sync", response_model=AnalysisResponse)
async def analyze_contract_sync(request: ContractAnalysisRequest):
    """
    Synchronous contract analysis (for small contracts).
    
    Use async endpoint for large contracts.
    """
    try:
        analysis_id = str(uuid.uuid4())
        
        # Perform analysis
        result = await perform_analysis_internal(analysis_id, request)
        
        return result
        
    except Exception as e:
        logger.error(f"Sync analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint
@app.post("/analyze/upload")
async def analyze_contract_file(
    file: UploadFile = File(...),
    regulations: List[str] = ["gdpr"],
    focus_areas: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = None
):
    """Analyze contract from uploaded file."""
    
    # Validate file type
    if not file.content_type.startswith('text/'):
        raise HTTPException(
            status_code=400, 
            detail="Only text files are supported"
        )
    
    try:
        # Read file content
        content = await file.read()
        contract_text = content.decode('utf-8')
        
        # Create analysis request
        request = ContractAnalysisRequest(
            contract_text=contract_text,
            contract_id=file.filename,
            regulations=regulations,
            focus_areas=focus_areas
        )
        
        # Start analysis
        analysis_id = str(uuid.uuid4())
        analysis_results[analysis_id] = AnalysisStatus(
            analysis_id=analysis_id,
            status="pending",
            progress=0,
            message="File uploaded, analysis queued"
        )
        
        background_tasks.add_task(perform_analysis, analysis_id, request)
        
        return {
            "analysis_id": analysis_id,
            "status": "accepted",
            "filename": file.filename,
            "size": len(contract_text)
        }
        
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File encoding not supported. Please use UTF-8."
        )
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail="File processing failed")

# Information endpoints
@app.get("/regulations")
async def list_regulations():
    """List available regulations."""
    return {
        "regulations": [
            {
                "id": "gdpr",
                "name": "General Data Protection Regulation",
                "requirements": len(regulations["gdpr"])
            },
            {
                "id": "ai_act", 
                "name": "EU AI Act",
                "requirements": len(regulations["ai_act"])
            },
            {
                "id": "ccpa",
                "name": "California Consumer Privacy Act", 
                "requirements": len(regulations["ccpa"])
            }
        ]
    }

@app.get("/regulations/{regulation_id}/requirements")
async def get_regulation_requirements(regulation_id: str):
    """Get requirements for specific regulation."""
    if regulation_id not in regulations:
        raise HTTPException(status_code=404, detail="Regulation not found")
    
    regulation = regulations[regulation_id]
    requirements = regulation.get_requirements()
    
    return {
        "regulation": regulation.name,
        "total_requirements": len(requirements),
        "requirements": [
            {
                "id": req_id,
                "description": req.description,
                "article_reference": req.article_reference,
                "mandatory": req.mandatory,
                "categories": list(req.categories)
            }
            for req_id, req in requirements.items()
        ]
    }

# Background task functions
async def perform_analysis(analysis_id: str, request: ContractAnalysisRequest):
    """Perform contract analysis in background."""
    try:
        # Update status
        analysis_results[analysis_id] = AnalysisStatus(
            analysis_id=analysis_id,
            status="processing",
            progress=10,
            message="Parsing contract"
        )
        
        result = await perform_analysis_internal(analysis_id, request)
        
        # Store completed result
        analysis_results[analysis_id] = result
        
    except Exception as e:
        logger.error(f"Analysis error for {analysis_id}: {e}")
        analysis_results[analysis_id] = AnalysisStatus(
            analysis_id=analysis_id,
            status="failed",
            progress=0,
            message=str(e)
        )

async def perform_analysis_internal(
    analysis_id: str, 
    request: ContractAnalysisRequest
) -> AnalysisResponse:
    """Internal analysis logic."""
    
    # Parse contract
    contract_id = request.contract_id or f"contract_{analysis_id[:8]}"
    parsed_contract = parser.parse(request.contract_text, contract_id)
    
    # Update progress if this is a background task
    if analysis_id in analysis_results and isinstance(analysis_results[analysis_id], AnalysisStatus):
        analysis_results[analysis_id].progress = 30
        analysis_results[analysis_id].message = "Verifying compliance"
    
    # Verify compliance for each regulation
    all_results = {}
    total_compliant = 0
    total_requirements = 0
    
    for reg_name in request.regulations:
        if reg_name not in regulations:
            continue
            
        regulation = regulations[reg_name]
        results = prover.verify_compliance(
            contract=parsed_contract,
            regulation=regulation,
            focus_areas=request.focus_areas
        )
        
        # Convert results to response format
        response_results = []
        for req_id, result in results.items():
            response_results.append(ComplianceResultResponse(
                requirement_id=result.requirement_id,
                requirement_description=result.requirement_description,
                status=result.status.value,
                confidence=result.confidence,
                issue=result.issue,
                suggestion=result.suggestion
            ))
            
            if result.compliant:
                total_compliant += 1
            total_requirements += 1
        
        all_results[reg_name] = response_results
    
    # Update progress
    if analysis_id in analysis_results and isinstance(analysis_results[analysis_id], AnalysisStatus):
        analysis_results[analysis_id].progress = 90
        analysis_results[analysis_id].message = "Finalizing results"
    
    # Calculate overall compliance rate
    overall_rate = (total_compliant / total_requirements * 100) if total_requirements > 0 else 0
    
    # Generate report
    report = prover.generate_compliance_report(
        contract=parsed_contract,
        regulation=regulations[request.regulations[0]],  # Use first regulation for report
        results={req.requirement_id: prover._create_basic_result(req.requirement_id, req.status == "compliant") 
                for reg_results in all_results.values() for req in reg_results}
    )
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        contract_id=contract_id,
        status="completed",
        compliance_results=all_results,
        overall_compliance_rate=overall_rate,
        timestamp=report.timestamp
    )

# Utility function for basic result creation
def create_basic_result(requirement_id: str, compliant: bool):
    """Create basic compliance result."""
    from neuro_symbolic_law.core.compliance_result import ComplianceResult, ComplianceStatus
    
    return ComplianceResult(
        requirement_id=requirement_id,
        requirement_description=f"Requirement {requirement_id}",
        status=ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT,
        confidence=0.8
    )

# Add the utility method to prover
prover._create_basic_result = create_basic_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)