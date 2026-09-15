from app.workflow.router import keyword_fallback_route


def test_route_visualization():
    assert keyword_fallback_route("can you plot the sales trend") == "visualization"


def test_route_ml_anomaly():
    assert keyword_fallback_route("are there any outliers in this data") == "ml_anomaly"


def test_route_data_analysis():
    assert keyword_fallback_route("what is the average salary") == "data_analysis"


def test_route_document_retrieval():
    assert keyword_fallback_route("what does the churn KPI mean") == "document_retrieval"


def test_route_general_fallback():
    assert keyword_fallback_route("hello there") == "general"
