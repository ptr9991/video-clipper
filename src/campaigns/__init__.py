"""Campaign profiles (Campeonato Dona 30K, etc.)."""

from src.campaigns.loader import list_campaigns, load_campaign
from src.campaigns.models import CampaignProfile, ValidationResult
from src.campaigns.validator import validate_campaign_export
from src.campaigns.copygen import build_platform_copy
from src.campaigns.apply import apply_campaign_to_project

__all__ = [
    "CampaignProfile",
    "ValidationResult",
    "list_campaigns",
    "load_campaign",
    "validate_campaign_export",
    "build_platform_copy",
    "apply_campaign_to_project",
]
