#!/usr/bin/env bash
set -e

MCP_URL="http://localhost:8000/mcp"

# Step 1: Initialize session
INIT_RESPONSE=$(curl -s -D - -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": { "name": "test-client", "version": "1.0" }
    }
  }')

SESSION_ID=$(echo "$INIT_RESPONSE" | grep -i "mcp-session-id:" | awk '{print $2}' | tr -d '\r')
echo "Session ID: $SESSION_ID"

# Step 2: Send email
curl -s -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "send_gmail_message",
      "arguments": {
        "user_google_email": "[EMAIL_1]",
        "to": "[EMAIL_1]",
        "subject": "Why don'\''t scientists trust atoms?",
        "body": "Because they make up everything! 😄\n\nHope this brightened your day!"
      }
    }
  }'
