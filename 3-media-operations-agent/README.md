# Multi-Agent Sports Analysis Labs

This repository contains a series of labs demonstrating how to build AI agents for automated sports video analysis and content compliance using Amazon Bedrock.

## Lab Structure

### Prerequisites
**00-prerequisites/** - Run this FIRST
- Installs all required Python packages
- Creates shared AWS resources (Knowledge Bases, DynamoDB tables, Lambda, Gateway)
- Stores configuration for all labs
- **Time:** 10-15 minutes

### Labs

**Lab 1: Sports Agent** (`1-create-an-sports-agent/`)
- Build a sports video analysis agent
- Use BDA for video understanding
- Query Knowledge Base and DynamoDB for missing information
- **Uses:** Sports KB, Players table (from prerequisites)

**Lab 2: News Agent** (`2-news-agent/`)
- Create a news content extraction agent
- Extract WHO, WHAT, WHEN, WHERE, WHY from videos
- **Uses:** News KB (from prerequisites)

**Lab 3: Sports Agent on Runtime** (`3-sports-agent-on-runtime/`)
- Deploy sports agent to AgentCore Runtime
- Production deployment with containerization
- **Uses:** Sports KB, Players table (from prerequisites)

**Lab 4: Multi-Agent Orchestration** (`4-multi-agent-for-sports-analysis/`)
- Build an orchestrator that coordinates multiple specialized agents
- Intelligent query routing and workflow management
- **Uses:** All KBs, all tables, Gateway (from prerequisites)

### Cleanup
**99-cleanup/** - Run this LAST
- Deletes all AWS resources created across all labs
- Removes AgentCore Runtimes, ECR repositories
- Clears SSM parameters and stored variables
- **Time:** 2-3 minutes

## Quick Start

```bash
# 1. Run prerequisites (once)
jupyter notebook blog/00-prerequisites/00-prerequisites.ipynb

# 2. Run any lab
jupyter notebook blog/1-create-an-sports-agent/create-an-sports-agent.ipynb

# 3. When done, cleanup
jupyter notebook blog/99-cleanup/99-cleanup.ipynb
```

## Shared Resources

The following resources are created once in prerequisites and reused across labs:

### Knowledge Bases
- **Sports KB** - Match information, venues, teams (Labs 1, 3, 4)
- **Compliance KB** - Content standards and rules (Lab 4)
- **News KB** - News articles and context (Labs 2, 4)
- **Films KB** - Movie and cast information (Lab 4)

### DynamoDB Tables
- **Players Table** - Player names, numbers, positions (Labs 1, 3, 4)
- **Cast Table** - Film cast members (Lab 4)

### AgentCore Infrastructure
- **Lambda Function** - 7 MCP tools for agent access
- **AgentCore Gateway** - MCP protocol server with JWT auth
- **Cognito User Pool** - Authentication for gateway

### Configuration Storage
- **SSM Parameters** - Gateway URLs, KB IDs, table names
- **Jupyter %store** - Cross-notebook variable sharing
- **config.json** - Backup configuration file

## Resource Naming Convention

All resources use consistent naming with account-specific suffixes:

```
sports-kb-{account_id[:6]}
compliance-kb-{account_id[:6]}
players-{account_id[:6]}
sports-tools-gateway-{account_id[:6]}
```

This prevents naming conflicts when multiple users run labs in the same account.

## Cost Optimization

**Shared Resources Approach:**
- Knowledge Bases created once, reused 4x → **75% time savings**
- DynamoDB tables created once, reused 3x → **67% time savings**
- Gateway created once, reused 2x → **50% time savings**

**Estimated Costs:**
- Knowledge Bases: ~$0.10/hour (OpenSearch Serverless)
- DynamoDB: ~$0.25/month (on-demand)
- Lambda: ~$0.20/million requests
- AgentCore Gateway: ~$0.01/1000 requests
- AgentCore Runtime: ~$0.10/hour when deployed

**Total for all labs:** ~$2-5 if completed in one session

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    00-prerequisites.ipynb                   │
│                                                             │
│  Creates:                                                   │
│  • 4 Knowledge Bases (sports, compliance, news, films)      │
│  • 2 DynamoDB Tables (players, cast)                        │
│  • 1 Lambda Function (7 MCP tools)                          │
│  • 1 AgentCore Gateway (with Cognito auth)                  │
│  • SSM Parameters for configuration                         │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬────────────────┐
         │               │               │                │
         ▼               ▼               ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────────┐
│   LAB 1     │  │   LAB 2     │  │  LAB 3   │  │    LAB 4     │
│   Sports    │  │   News      │  │  Runtime │  │ Multi-Agent  │
│   Agent     │  │   Agent     │  │  Deploy  │  │ Orchestrator │
└─────────────┘  └─────────────┘  └──────────┘  └──────────────┘
         │               │               │                │
         └───────────────┴───────────────┴────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────┐
         │       99-cleanup.ipynb                │
         │                                       │
         │  Deletes all resources                │
         └───────────────────────────────────────┘
```

## Troubleshooting

### "Knowledge Base not found"
- Run `00-prerequisites.ipynb` first
- Check that `%store -r sports_kb_id` returns a value
- Verify KB exists in AWS Console

### "Table does not exist"
- Run `00-prerequisites.ipynb` first
- Check DynamoDB console for tables with your account suffix
- Verify SSM parameters exist: `/sports-agent/*`

### "Gateway authentication failed"
- Check Cognito User Pool exists
- Verify client ID and secret in SSM
- Ensure gateway is in READY state

### "Runtime deployment failed"
- Check CloudWatch logs for error details
- Verify IAM role has required permissions
- Ensure ECR repository was created

## Additional Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AgentCore Runtime Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore-runtime.html)
- [Strands Agents Documentation](https://strandsagents.com/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review CloudWatch logs for detailed error messages
3. Verify all prerequisites completed successfully
4. Check AWS service quotas and limits
