"""OpenAI Agents SDK tools that expose the existing deterministic library."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from agents import function_tool

from lib import (
    ApifyClient,
    available_models,
    available_templates,
    brand_logo,
    card,
    fetch_post,
    illustrate,
    illustrate_set,
    parse_linkedin_url,
    publish,
    quote_card,
    refine,
    repost,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _manual_read_message() -> dict[str, str]:
    return {
        "mode": "manual",
        "message": (
            "APIFY_TOKEN is not configured. Ask the user to paste the LinkedIn "
            "post/comment text instead of fabricating fetched data."
        ),
    }


@function_tool
def parse_linkedin_url_tool(url: str) -> str:
    """Parse a LinkedIn post/comment URL into the canonical URNs used by the library."""
    try:
        return _json(parse_linkedin_url(url))
    except Exception as exc:
        return _json(
            {
                "mode": "error",
                "message": f"Invalid LinkedIn URL: {type(exc).__name__}: {exc}",
            }
        )


@function_tool
def get_linkedin_post(url: str, force_refresh: bool = False) -> str:
    """Fetch a LinkedIn post by URL through the existing Apify read layer."""
    if not os.getenv("APIFY_TOKEN"):
        return _json(_manual_read_message())
    try:
        return _json(
            fetch_post(url, force_refresh=force_refresh) or _manual_read_message()
        )
    except Exception as exc:
        return _json(
            {
                "mode": "error",
                "message": (
                    "LinkedIn post could not be fetched: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


@function_tool
def get_linkedin_comments(
    post_id: str,
    max_items: int = 20,
    sort_order: str = "most relevant",
    force_refresh: bool = False,
) -> str:
    """Fetch a LinkedIn post's comments and replies through Apify."""
    if not os.getenv("APIFY_TOKEN"):
        return _json(_manual_read_message())
    try:
        rows = ApifyClient().fetch_post_comments(
            post_id=post_id,
            max_items=max_items,
            sort_order=sort_order,
            force_refresh=force_refresh,
        )
        return _json({"mode": "apify", "comments": rows})
    except Exception as exc:
        return _json(
            {
                "mode": "error",
                "message": (
                    "Comments could not be fetched: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


@function_tool
def get_recent_linkedin_comments(
    username: str,
    result_limit: int = 30,
    force_refresh: bool = False,
) -> str:
    """Fetch a user's recent LinkedIn comments through Apify."""
    if not os.getenv("APIFY_TOKEN"):
        return _json(_manual_read_message())
    try:
        rows = ApifyClient().fetch_user_recent_comments(
            username=username,
            result_limit=result_limit,
            force_refresh=force_refresh,
        )
        return _json({"mode": "apify", "comments": rows})
    except Exception as exc:
        return _json(
            {
                "mode": "error",
                "message": (
                    "Recent comments could not be fetched: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


@function_tool
def get_linkedin_engagers(
    post_url: str,
    max_items: int = 50,
    types: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> str:
    """Fetch people who liked, commented on, or reshared a LinkedIn post."""
    if not os.getenv("APIFY_TOKEN"):
        return _json(_manual_read_message())
    try:
        rows = ApifyClient().fetch_post_engagers(
            post_url=post_url,
            max_items=max_items,
            types=tuple(types or ("likers", "commenters")),
            force_refresh=force_refresh,
        )
        return _json({"mode": "apify", "engagers": rows})
    except Exception as exc:
        return _json(
            {
                "mode": "error",
                "message": (
                    "Engagers could not be fetched: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


@function_tool
def get_linkedin_image_models() -> str:
    """List available Pixfaro image models when image generation is configured."""
    try:
        return _json({"models": available_models()})
    except Exception as exc:
        return _json(
            {
                "mode": "error",
                "message": (
                    "Image model catalog unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


@function_tool
def get_linkedin_image_templates() -> str:
    """List available Pixfaro design templates when image rendering is configured."""
    try:
        return _json({"templates": available_templates()})
    except Exception as exc:
        return _json(
            {
                "mode": "error",
                "message": (
                    "Image template catalog unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )


@function_tool(needs_approval=True)
def publish_linkedin_post(
    draft_text: str,
    scheduled_time: Optional[str] = None,
    media_urls: Optional[list[str]] = None,
) -> str:
    """Publish or schedule an approved LinkedIn post via the configured backend."""
    result = publish(
        kind="post",
        draft_text=draft_text,
        target_url="https://www.linkedin.com/post/new/",
        platforms=[{"platform": "linkedin", "platformId": os.getenv("LINKEDIN_PLATFORM_ID")}],
        scheduled_time=scheduled_time,
        media_urls=media_urls,
    )
    return _json(
        result
        or {"mode": "error", "message": "Publishing backend returned no result."}
    )


@function_tool(needs_approval=True)
def publish_linkedin_comment(
    post_url: str,
    post_urn: str,
    draft_text: str,
    platform_id: Optional[str] = None,
    reaction_type: Optional[str] = None,
) -> str:
    """Publish an approved top-level LinkedIn comment via the configured backend."""
    result = publish(
        kind="comment",
        draft_text=draft_text,
        target_url=post_url,
        post_urn=post_urn,
        platform_id=platform_id or os.getenv("LINKEDIN_PLATFORM_ID"),
        reaction_type=reaction_type,
    )
    return _json(
        result
        or {"mode": "error", "message": "Publishing backend returned no result."}
    )


@function_tool(needs_approval=True)
def publish_linkedin_reply(
    post_url: str,
    post_urn: str,
    parent_comment: str,
    draft_text: str,
    platform_id: Optional[str] = None,
    reaction_type: Optional[str] = None,
) -> str:
    """Publish an approved LinkedIn reply using the supplied parent comment URN."""
    result = publish(
        kind="reply",
        draft_text=draft_text,
        target_url=post_url,
        post_urn=post_urn,
        parent_comment=parent_comment,
        platform_id=platform_id or os.getenv("LINKEDIN_PLATFORM_ID"),
        reaction_type=reaction_type,
    )
    return _json(
        result
        or {"mode": "error", "message": "Publishing backend returned no result."}
    )


@function_tool(needs_approval=True)
def reshare_linkedin_post(
    post_url: str,
    commentary: Optional[str] = None,
    parent_urn: Optional[str] = None,
    platform_id: Optional[str] = None,
) -> str:
    """Reshare an approved LinkedIn post with optional commentary."""
    result = repost(
        post_url=post_url,
        commentary=commentary,
        parent=parent_urn,
        platform_id=platform_id or os.getenv("LINKEDIN_PLATFORM_ID"),
    )
    return _json(
        result
        or {"mode": "error", "message": "Reshare backend returned no result."}
    )


@function_tool(needs_approval=True)
def cancel_linkedin_scheduled_post(post_group_id: str) -> str:
    """Cancel an approved/scheduled post before it goes live."""
    from lib import unpublish

    result = unpublish(post_group_id=post_group_id)
    return _json(
        result
        or {"mode": "error", "message": "No cancellation result returned."}
    )


@function_tool(needs_approval=True)
def generate_linkedin_image(
    prompt: str,
    kind: str = "post",
    model: Optional[str] = None,
    resolution: str = "1K",
) -> str:
    """Generate a LinkedIn visual through the existing Pixfaro integration."""
    return _json(
        illustrate(prompt, kind=kind, model=model, resolution=resolution)
    )


@function_tool(needs_approval=True)
def generate_linkedin_image_set(
    prompts: list[str],
    kind: str = "wide",
    model: Optional[str] = None,
    resolution: str = "1K",
) -> str:
    """Generate a 2-10 image LinkedIn grid through Pixfaro."""
    return _json(
        illustrate_set(
            prompts,
            kind=kind,
            model=model,
            resolution=resolution,
        )
    )


@function_tool(needs_approval=True)
def refine_linkedin_image(
    image_id: str,
    instruction: str,
    model: Optional[str] = None,
) -> str:
    """Refine a previously generated Pixfaro image."""
    return _json(refine(image_id, instruction, model=model))


@function_tool(needs_approval=True)
def render_linkedin_quote_card(
    quote: str,
    name: Optional[str] = None,
    handle: Optional[str] = None,
    size: str = "1:1",
) -> str:
    """Render a crisp, typeset LinkedIn quote card with Pixfaro."""
    return _json(
        quote_card(
            quote,
            name=name,
            handle=handle,
            size=size,
        )
    )


@function_tool(needs_approval=True)
def render_linkedin_card(
    template: str,
    slots: dict[str, Any],
    size: Optional[str] = None,
    style: Optional[Any] = None,
) -> str:
    """Render a generic Pixfaro design template."""
    return _json(
        card(
            template,
            slots,
            size=size,
            style=style,
        )
    )


@function_tool(needs_approval=True)
def upload_linkedin_brand_logo(
    path: str,
    name: Optional[str] = None,
) -> str:
    """Upload a brand logo from the repository assets area to Pixfaro."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        candidate = candidate.resolve()
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return _json(
            {
                "mode": "error",
                "message": "Logo path must stay inside the linkedin-agent repository.",
            }
        )
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return _json(
            {
                "mode": "error",
                "message": "Logo must be a PNG, JPG, JPEG, or WEBP file.",
            }
        )
    if not candidate.is_file():
        return _json(
            {
                "mode": "error",
                "message": f"Logo file does not exist: {candidate}",
            }
        )
    if candidate.stat().st_size > 1_000_000:
        return _json(
            {
                "mode": "error",
                "message": "Logo file is larger than the 1 MB Pixfaro upload limit.",
            }
        )
    return _json(brand_logo(str(candidate), name=name))


READ_TOOLS = [
    parse_linkedin_url_tool,
    get_linkedin_post,
    get_linkedin_comments,
    get_recent_linkedin_comments,
    get_linkedin_engagers,
    get_linkedin_image_models,
    get_linkedin_image_templates,
]

WRITE_TOOLS = [
    publish_linkedin_post,
    publish_linkedin_comment,
    publish_linkedin_reply,
    reshare_linkedin_post,
    cancel_linkedin_scheduled_post,
    generate_linkedin_image,
    generate_linkedin_image_set,
    refine_linkedin_image,
    render_linkedin_quote_card,
    render_linkedin_card,
    upload_linkedin_brand_logo,
]

ALL_TOOLS = READ_TOOLS + WRITE_TOOLS
