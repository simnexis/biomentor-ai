"""Synthetic demo fixtures.

IMPORTANT: these are **not real publications**. They are short, textbook-level paragraphs written
for this project so that the pipeline, tests and evaluation code can run offline. They have no PMID
or DOI, are titled "[Synthetic fixture] ...", and carry ``source_type="synthetic_fixture"`` so the
UI always labels them as demo data. For real literature use ``scripts/fetch_literature.py``.

The accompanying benchmark is likewise synthetic: it only proves the evaluation code works. It says
nothing about performance on real biomedical literature.
"""

from __future__ import annotations

from biomentor.schemas import SOURCE_SYNTHETIC, Document

SOURCE_NAME = "BioMentor synthetic test fixture"
FIXTURE_LICENSE = "CC0-1.0 (written for this project; not derived from any publication)"
FIXTURE_DATE = "2026-01-01T00:00:00+00:00"

# (id, title, abstract)
_FIXTURES: list[tuple[str, str, str]] = [
    ("syn-t2d-01", "Insulin resistance and beta-cell function in type 2 diabetes",
     "Type 2 diabetes mellitus is characterised by reduced responsiveness of muscle, liver and fat tissue to "
     "insulin, together with a progressive decline in the ability of pancreatic beta cells to secrete enough "
     "insulin. When cells respond poorly to insulin, less glucose is taken up from the blood and the liver "
     "continues to release glucose, so blood glucose concentrations rise. Over time the beta cells must work "
     "harder, and their capacity to compensate can fail. This text describes the textbook two-part model of "
     "impaired insulin action plus impaired insulin secretion."),
    ("syn-t2d-02", "Physical activity, body weight and glycaemic control",
     "Regular physical activity increases glucose uptake by skeletal muscle through insulin-dependent and "
     "insulin-independent pathways. Body weight reduction in people with excess adiposity is associated with "
     "improved insulin sensitivity. Dietary patterns that limit rapidly absorbed carbohydrates produce smaller "
     "rises in blood glucose after meals. This synthetic summary does not report any trial and gives no clinical "
     "recommendations."),
    ("syn-t2d-03", "Chronic hyperglycaemia and long-term vascular complications",
     "Persistently elevated blood glucose can damage small blood vessels in the retina, kidneys and peripheral "
     "nerves, leading to retinopathy, nephropathy and neuropathy. Mechanisms proposed in textbooks include "
     "accumulation of advanced glycation end products and increased oxidative stress in vascular cells. Large "
     "blood vessels are also affected, which increases the risk of cardiovascular disease."),
    ("syn-mal-01", "Life cycle of Plasmodium parasites in humans and mosquitoes",
     "Malaria is caused by Plasmodium parasites transmitted by the bite of infected female Anopheles mosquitoes. "
     "Sporozoites travel to the liver and invade hepatocytes, where they multiply and are released as "
     "merozoites. Merozoites then invade red blood cells and multiply again; the bursting of infected red blood "
     "cells releases more merozoites and is associated with the cyclical fever characteristic of the disease. "
     "Some parasites develop into gametocytes that can be taken up by a mosquito."),
    ("syn-mal-02", "Bed nets and indoor spraying as vector control",
     "Vector control reduces contact between people and mosquitoes. Insecticide-treated bed nets act as both a "
     "physical barrier and a chemical one that kills or repels mosquitoes that land on them. Indoor residual "
     "spraying treats the inner walls of houses with insecticide. Insecticide resistance in mosquito "
     "populations is a recognised challenge for these methods."),
    ("syn-mal-03", "Sickle cell trait and susceptibility to severe malaria",
     "Individuals who carry one copy of the sickle haemoglobin variant, known as sickle cell trait, are "
     "generally less likely to develop severe malaria than individuals without the variant. Proposed "
     "explanations include impaired parasite growth inside red blood cells that contain sickle haemoglobin and "
     "enhanced removal of infected cells. This textbook-level association is a classic example of balanced "
     "selection."),
    ("syn-asm-01", "Airway inflammation and narrowing in asthma",
     "Asthma is a chronic inflammatory disease of the airways. Exposure to triggers such as allergens or viral "
     "infections can cause airway smooth muscle to contract, the airway lining to swell, and mucus production "
     "to increase; together these narrow the airways and make breathing difficult. Eosinophils, mast cells and "
     "T helper 2 lymphocytes are among the immune cells commonly implicated in allergic asthma."),
    ("syn-asm-02", "Mechanisms of bronchodilators and inhaled corticosteroids",
     "Short-acting beta-2 agonists relax airway smooth muscle by activating beta-2 adrenergic receptors, which "
     "widens the airways within minutes. Inhaled corticosteroids reduce airway inflammation by altering gene "
     "expression in immune and airway cells, an effect that develops over days to weeks. This synthetic "
     "description explains drug mechanisms only and is not a treatment guideline."),
    ("syn-asm-03", "Allergen sensitisation and immunoglobulin E in allergic asthma",
     "In allergic asthma, exposure to an allergen can lead to production of allergen-specific immunoglobulin E "
     "(IgE). IgE binds to receptors on mast cells; later allergen exposure cross-links the bound IgE and "
     "triggers release of histamine and other mediators, producing rapid airway narrowing."),
    ("syn-tb-01", "Mycobacterium tuberculosis infection and granuloma formation",
     "Tuberculosis is caused by Mycobacterium tuberculosis, which spreads through airborne droplets. After "
     "inhalation the bacteria are engulfed by alveolar macrophages, in which they can survive. The immune "
     "system may wall off infected macrophages within structures called granulomas, which can contain the "
     "infection for long periods without clearing it completely."),
    ("syn-tb-02", "Why tuberculosis therapy combines several antibiotics",
     "Mycobacterium tuberculosis grows slowly and can exist in different metabolic states. Treatment regimens "
     "combine several antibiotics for months because a single drug would allow resistant bacteria to emerge "
     "and because different drugs act on different bacterial populations. Drug-resistant tuberculosis arises "
     "when treatment is incomplete or inappropriate."),
    ("syn-tb-03", "BCG vaccination and tuberculosis prevention",
     "The BCG vaccine is derived from a weakened strain of Mycobacterium bovis. It provides protection against "
     "severe forms of tuberculosis in young children, such as tuberculous meningitis, while its protection "
     "against pulmonary tuberculosis in adults is variable."),
]


def demo_documents() -> list[Document]:
    return [
        Document(
            document_id=doc_id,
            title=f"[Synthetic fixture] {title}",
            abstract=abstract,
            source_name=SOURCE_NAME,
            source_type=SOURCE_SYNTHETIC,
            pmid=None,
            doi=None,
            authors=None,
            journal=None,
            publication_date=None,
            publication_year=None,
            article_type="Synthetic test fixture",
            source_url=None,
            license=FIXTURE_LICENSE,
            retrieved_at=FIXTURE_DATE,
        )
        for doc_id, title, abstract in _FIXTURES
    ]


# (query, topic, {document_id: relevance grade 1-2}, note). Several queries deliberately avoid the
# fixtures' own vocabulary so that keyword and semantic retrieval can behave differently.
_BENCHMARK: list[tuple[str, str, dict[str, int], str]] = [
    ("insulin resistance beta cells", "type 2 diabetes", {"syn-t2d-01": 2}, "keyword-overlap query"),
    ("why do cells stop responding to the hormone that controls blood sugar", "type 2 diabetes", {"syn-t2d-01": 2}, "paraphrase, little lexical overlap"),
    ("how does exercise help muscles take up glucose", "type 2 diabetes", {"syn-t2d-02": 2}, "paraphrase"),
    ("damage to small blood vessels from high blood sugar", "type 2 diabetes", {"syn-t2d-03": 2}, "paraphrase"),
    ("Plasmodium life cycle liver merozoites", "malaria", {"syn-mal-01": 2}, "keyword-overlap query"),
    ("how does a mosquito bite lead to parasites in the blood", "malaria", {"syn-mal-01": 2}, "paraphrase"),
    ("protection against malaria by a haemoglobin variant", "malaria", {"syn-mal-03": 2}, "paraphrase"),
    ("insecticide treated nets to prevent mosquito bites", "malaria", {"syn-mal-02": 2}, "keyword-overlap query"),
    ("which immune cells cause airway narrowing in allergic asthma", "asthma", {"syn-asm-01": 2, "syn-asm-03": 1}, "two relevant documents, graded"),
    ("histamine release from mast cells after allergen exposure", "asthma", {"syn-asm-03": 2, "syn-asm-01": 1}, "two relevant documents, graded"),
    ("medicines that open the airways quickly versus reduce inflammation slowly", "asthma", {"syn-asm-02": 2}, "paraphrase"),
    ("macrophages walling off tuberculosis bacteria", "tuberculosis", {"syn-tb-01": 2}, "paraphrase"),
    ("why do tuberculosis patients need several antibiotics", "tuberculosis", {"syn-tb-02": 2}, "keyword-overlap query"),
    ("vaccine protection against tuberculosis in children", "tuberculosis", {"syn-tb-03": 2}, "keyword-overlap query"),
]


def demo_benchmark_records() -> list[dict]:
    records = []
    for i, (query, topic, grades, note) in enumerate(_BENCHMARK, start=1):
        records.append(
            {
                "question_id": f"syn-q{i:02d}",
                "query": query,
                "topic": topic,
                "relevant_document_ids": list(grades),
                "relevant_passage_ids": [f"{doc_id}-p000" for doc_id in grades],
                "relevance_grades": grades,
                "annotation_notes": note,
                "provenance": "SYNTHETIC fixture benchmark written with the synthetic fixtures; "
                "pipeline smoke test only - not a scientific benchmark.",
            }
        )
    return records
