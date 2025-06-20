
# backend/api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .agents import execute_agent_workflow, ExecutionStatus
import logging
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime
from asgiref.sync import async_to_sync

# Configure logging
logger = logging.getLogger(__name__)

# MongoDB setup
mongo_client = None
db = None
collection = None

try:
    load_dotenv()
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI not set.")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[os.getenv("MONGODB_DB_NAME", "hackathon")]
    collection = db['hackathon']
    logger.debug("Connected to MongoDB.")
except Exception as e:
    logger.error("Failed to connect to MongoDB: %s", e)

class StudentProfileView(APIView):
    @async_to_sync
    async def post(self, request):
        logger.debug("Received request data: %s", request.POST)
        logger.debug("Received files: %s", request.FILES)

        aspirations = request.POST.get('aspirations', '{}')
        profile_data = request.POST.get('profile', '{}')
        resume_file = request.FILES.get('resume')

        try:
            # Parse input data
            aspirations = json.loads(aspirations) if isinstance(aspirations, str) else aspirations
            profile_data = json.loads(profile_data) if isinstance(profile_data, str) else profile_data
            logger.debug("Processed aspirations: %s, profile_data: %s", aspirations, profile_data)

            # Execute agent workflow
            result = await execute_agent_workflow(aspirations, profile_data, resume_file, async_execution=True)

            # Handle workflow failure
            if result.status != ExecutionStatus.COMPLETED:
                logger.warning("Agent workflow failed: %s", result.error)
                return Response(
                    {
                        "error": result.error or "Failed to process request",
                        "details": str(result.error),
                        "partial_result": result.result
                    },
                    status=status.HTTP_200_OK  # Use 200 to inspect partial results
                )

            # Store result in MongoDB (optional)
            if collection is not None:
                try:
                    result_doc = {
                        "aspirations": result.result.get("aspirations", {}),
                        "profile_score": result.result.get("profile_score", {}),
                        "market_benchmarks": result.result.get("market_benchmarks", {}),
                        "gap_score": result.result.get("gap_score", 0),
                        "gap_details": result.result.get("gap_details", {}),
                        "micro_okrs": result.result.get("micro_okrs", []),
                        "report": result.result.get("report", {}),
                        "timestamp": datetime.datetime.now()
                    }
                    collection.insert_one(result_doc)
                    logger.debug("Result stored in MongoDB.")
                except Exception as mongo_error:
                    logger.warning("MongoDB storage failed, continuing: %s", mongo_error)

            return Response(result.result, status=status.HTTP_200_OK)
        except json.JSONDecodeError as e:
            logger.error("JSON DecodeError: %s", e)
            return Response({"error": "Invalid JSON data"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            return Response(
                {"error": "An unexpected error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )