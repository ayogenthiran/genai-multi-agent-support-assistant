"""System prompts and chat prompt templates used by support agents."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# Explains the assistant's supported demo capabilities for general queries.
GENERAL_CAPABILITY_REPLY = (
    "Hello. I am a customer support assistant for support executives. "
    "I can look up customer profiles and support ticket history from the customer database, "
    "answer policy questions using uploaded company policy documents, "
    "and combine both when customer data and policy guidance are needed."
)

# Routes ambiguous customer support queries to the appropriate specialist agent.
SUPERVISOR_ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route customer support queries for a support executive assistant.\n"
            "Return exactly one label: sql, rag, both, or general.\n\n"
            "Routing rules:\n"
            "- sql: customer/profile/ticket/order/account/history lookups that only need customer database data\n"
            "- rag: policy/refund/warranty/cancellation/document questions that only need policy documents\n"
            "- both: questions that mention a customer and also ask for policy guidance, eligibility, refund, warranty, cancellation, or what action to take\n"
            "- general: greetings, capability questions, thanks, or small talk\n\n"
            "Examples:\n"
            "Query: Give me Ema Johnson's profile and ticket history\n"
            "Label: sql\n"
            "Query: What is the current refund policy?\n"
            "Label: rag\n"
            "Query: Can Ema Johnson get a refund based on her support history and the refund policy?\n"
            "Label: both\n"
            "Query: Hi, what can you do?\n"
            "Label: general\n\n"
            "Reply with only the label.",
        ),
        ("human", "{query}"),
    ]
)

# Extracts a customer full name only when simple name heuristics are inconclusive.
SQL_CUSTOMER_NAME_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the customer full name from the support query. "
            "If no customer name is present, reply with NONE.",
        ),
        ("human", "{query}"),
    ]
)

# Rewrites natural-language policy questions into concise semantic search queries.
POLICY_QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite customer support questions into concise policy-focused "
            "search queries for retrieving company policy documents.\n"
            "Preserve the original intent. Do not answer the question.\n"
            "Return only the rewritten search query.",
        ),
        ("human", "{query}"),
    ]
)

# Generates grounded answers using only retrieved policy excerpts.
POLICY_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a customer support policy assistant helping a support executive.\n"
            "Answer ONLY using the provided policy excerpts.\n"
            "Do not treat template placeholders such as [DATE], [WEBSITE_NAME], "
            "or [COMPANY_INFORMATION] as real company facts.\n"
            "If the excerpts do not contain enough information, clearly state that "
            "the uploaded policy documents do not contain enough information.\n\n"
            "Structure your response with these sections:\n"
            "1. Direct answer\n"
            "2. Key policy conditions\n"
            "3. Recommended next step for the support executive\n"
            "4. Sources used (file name and page number for each excerpt referenced)",
        ),
        (
            "human",
            "Policy excerpts:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

# Formats the RAG agent's retrieved answer and supporting excerpts for synthesis.
RAG_LOOKUP_RESPONSE_TEMPLATE = (
    "Policy answer:\n"
    "{formatted_answer}\n\n"
    "Supporting policy excerpts:\n"
    "{passages}"
)

# Synthesizes specialist outputs into one executive-ready support response.
RESPONSE_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write concise, professional replies for customer support executives.\n"
            "Use the provided customer data and/or policy guidance when available.\n"
            "Be practical, accurate, and clear.\n"
            "Explicitly state whether the answer is based on customer data, "
            "policy documents, or both.\n"
            "When customer data is provided, summarize it in natural language "
            "with short paragraphs and bullets. Never expose raw JSON.\n"
            "If customer data is labeled Relevant support history, preserve that "
            "label and do not call the returned filtered tickets the customer's "
            "total ticket count.\n"
            "For SQL/customer-data responses, do not add recommendations, next actions, "
            "or prioritization advice unless the user explicitly asks for them.\n"
            "Do not invent facts that are not supported by the provided context.",
        ),
        (
            "human",
            "User question:\n{query}\n\n"
            "Routing label: {route}\n"
            "Expected source basis: {source}\n\n"
            "Customer data context:\n{sql_result}\n\n"
            "Policy context:\n{rag_context}",
        ),
    ]
)
