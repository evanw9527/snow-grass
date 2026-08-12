from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status

from snow_grass.pet.schemas import PetRespondRequest, PetRespondResponse
from snow_grass.pet.service import PetResponseService
from snow_grass.providers.base import ProviderError

pet_router = APIRouter(prefix="/api/v1/pet", tags=["pet"])


@pet_router.post("/respond", response_model=PetRespondResponse)
async def respond(request: Request, payload: PetRespondRequest) -> PetRespondResponse:
    service = cast(PetResponseService, request.app.state.pet_service)
    try:
        return await service.respond(payload)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
