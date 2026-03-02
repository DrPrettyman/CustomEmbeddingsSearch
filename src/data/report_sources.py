"""ESG report sources for passage extraction.

Combines ESGBench document sources with additional publicly available
sustainability reports and CDP responses. All URLs point to freely
accessible PDFs on company websites.
"""

# ESGBench source documents (from docs_seed.csv)
ESGBENCH_REPORTS = {
    "apple_esg_2024": {
        "url": "https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2024.pdf",
        "company": "Apple Inc",
        "industry": "Technology",
        "country": "US",
        "doc_type": "esg_report",
    },
    "microsoft_esg_2024": {
        "url": "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/Microsoft-2024-Environmental-Sustainability-Report.pdf",
        "company": "Microsoft Corp",
        "industry": "Technology",
        "country": "US",
        "doc_type": "esg_report",
    },
    "exxon_climate_2024": {
        "url": "https://corporate.exxonmobil.com/-/media/global/files/advancing-climate-solutions/2024/2024-advancing-climate-solutions-report.pdf",
        "company": "Exxon Mobil Corporation",
        "industry": "Energy",
        "country": "US",
        "doc_type": "climate_report",
    },
    "jpmorgan_climate_2024": {
        "url": "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/Climate-Report-2024.pdf",
        "company": "JPMorgan Chase & Co.",
        "industry": "Financial Services",
        "country": "US",
        "doc_type": "climate_report",
    },
    "cocacola_env_2024": {
        "url": "https://www.coca-colacompany.com/content/dam/company/us/en/reports/2024-environmental-update/2024-environmental-update.pdf",
        "company": "The Coca-Cola Company",
        "industry": "Consumer Goods",
        "country": "US",
        "doc_type": "esg_report",
    },
    "unilever_annual_2024": {
        "url": "https://www.unilever.com/files/unilever-annual-report-and-accounts-2024.pdf",
        "company": "Unilever Plc",
        "industry": "Consumer Goods",
        "country": "UK/NL",
        "doc_type": "annual_report",
    },
    "samsung_sustainability_2024": {
        "url": "https://www.samsung.com/global/sustainability/media/pdf/Samsung_Electronics_Sustainability_Report_2024_ENG.pdf",
        "company": "Samsung Electronics Co. Ltd.",
        "industry": "Technology",
        "country": "KR",
        "doc_type": "sustainability_report",
    },
    "safaricom_sustainability_2024": {
        "url": "https://www.safaricom.co.ke/images/Downloads/Safaricom-Sustainable-Report-2024-compressed.pdf",
        "company": "Safaricom Plc",
        "industry": "Telecom",
        "country": "KE",
        "doc_type": "sustainability_report",
    },
}

# Additional sustainability reports for broader coverage
ADDITIONAL_REPORTS = {
    "shell_sustainability_2024": {
        "url": "https://reports.shell.com/sustainability-report/2024/_assets/downloads/shell-sustainability-report-2024.pdf",
        "company": "Shell plc",
        "industry": "Energy",
        "country": "UK/NL",
        "doc_type": "sustainability_report",
    },
    "nestle_csv_2024": {
        "url": "https://www.nestle.com/sites/default/files/2025-02/creating-shared-value-nestle-2024.pdf",
        "company": "Nestle SA",
        "industry": "Consumer Goods",
        "country": "CH",
        "doc_type": "sustainability_report",
    },
}

# CDP questionnaire responses (company-published)
CDP_REPORTS = {
    "apple_cdp_2024": {
        "url": "https://www.apple.com/environment/pdf/Apple_CDP-Climate-Change-Questionnaire_2024.pdf",
        "company": "Apple Inc",
        "industry": "Technology",
        "country": "US",
        "doc_type": "cdp_response",
    },
    "bp_cdp_2024": {
        "url": "https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/sustainability/group-reports/bp-cdp-climate-change-questionnaire-2024.pdf",
        "company": "BP plc",
        "industry": "Energy",
        "country": "UK",
        "doc_type": "cdp_response",
    },
    "oracle_cdp_2024": {
        "url": "https://www.oracle.com/a/ocom/docs/cdp-climate-change-questionnaire-2024.pdf",
        "company": "Oracle Corporation",
        "industry": "Technology",
        "country": "US",
        "doc_type": "cdp_response",
    },
    "cisco_cdp_2024": {
        "url": "https://www.cisco.com/c/dam/m/en_us/about/csr/esg-hub/_pdf/2024-Cisco-CDP-Climate-Change-Response.pdf",
        "company": "Cisco Systems",
        "industry": "Technology",
        "country": "US",
        "doc_type": "cdp_response",
    },
}

ALL_REPORTS = {**ESGBENCH_REPORTS, **ADDITIONAL_REPORTS, **CDP_REPORTS}
