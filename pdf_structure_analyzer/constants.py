# pdf_structure_analyzer/constants.py
COMMON_HEADINGS = [
    "Objective", "Material Required", "Method", "Method of Construction",
    "Procedure", "Demonstration", "Observation", "Application",
    "Summary", "Background", "Introduction", "Conclusion", "References",
    "Result", "Discussion", "Theory", "Aim", "Apparatus", "Experiment",
    "Activity", "Steps", "Steps Involved", "Note", "Precautions"
]

NON_HEADING_KEYWORDS = {
    "table of contents", "index", "copyright notice",
    "revision history", "version history", "contents"
}