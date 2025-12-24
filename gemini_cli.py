#!/usr/bin/env python3
"""
Gemini 2.0 Flash CLI
A simple command-line interface for the Gemini 2.0 Flash API.
"""

import argparse
import json
import sys
from typing import Dict, Any

import requests

API_KEY = "AIzaSyDiET0PoOJSlWDJELnbi7JTurC5GlHrN64"
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.0-flash:generateContent"
)


def generate_content(prompt: str, debug: bool = False) -> str:
    """
    Send a prompt to Gemini 2.0 Flash and return the generated response.
    
    Args:
        prompt: The text prompt to send to Gemini
        debug: If True, print the raw API response
        
    Returns:
        The generated text response from Gemini
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
        KeyError: If the response format is unexpected
    """
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(
            f"{ENDPOINT}?key={API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if debug:
            print("=" * 60)
            print("RAW API RESPONSE:")
            print("=" * 60)
            print(json.dumps(data, indent=2))
            print("=" * 60)
            print()

        # Extract the generated text from the response
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.HTTPError as e:
        # Handle rate limit errors specially
        if e.response.status_code == 429:
            error_data = e.response.json() if e.response.text else {}
            error_message = error_data.get('error', {}).get('message', 'Rate limit exceeded')
            
            print(f"\n⚠️  Rate Limit Error (429)", file=sys.stderr)
            print(f"─" * 60, file=sys.stderr)
            print(f"{error_message}", file=sys.stderr)
            print(f"─" * 60, file=sys.stderr)
            
            # Extract retry delay if available
            retry_info = None
            for detail in error_data.get('error', {}).get('details', []):
                if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                    retry_info = detail.get('retryDelay', '')
                    break
            
            if retry_info:
                print(f"\n💡 Suggested action: Wait {retry_info} and try again", file=sys.stderr)
            else:
                print(f"\n💡 Suggested action: Wait a few moments and try again", file=sys.stderr)
            
            print(f"📚 Learn more: https://ai.google.dev/gemini-api/docs/rate-limits", file=sys.stderr)
            
            if debug:
                print(f"\nFull error response:", file=sys.stderr)
                print(json.dumps(error_data, indent=2), file=sys.stderr)
        else:
            print(f"HTTP Error: {e}", file=sys.stderr)
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}", file=sys.stderr)
        raise
    except KeyError as e:
        print(f"Unexpected response format: {e}", file=sys.stderr)
        print(f"Response data: {json.dumps(data, indent=2)}", file=sys.stderr)
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI for Gemini 2.0 Flash API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemini_cli.py "What is the capital of France?"
  python gemini_cli.py "Write a haiku about coding" --debug
        """
    )
    parser.add_argument(
        "prompt",
        help="The prompt to send to Gemini"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the raw API response payload"
    )

    args = parser.parse_args()

    try:
        output = generate_content(args.prompt, debug=args.debug)
        
        print("=" * 60)
        print("GEMINI RESPONSE:")
        print("=" * 60)
        print(output)
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()