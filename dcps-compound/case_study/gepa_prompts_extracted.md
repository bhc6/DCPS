# GEPA / GEPA-MERGE Optimized Prompts (gpt-4.1-mini)

Extracted from W&B project `awesome-prompt/GEPA` for the DCPS case study.

Source of prompt text: the `prompts/table` logged artifact of each **final_eval** run (columns: `predictor_name`, `instructions`, `signature`, `fields`). This table holds the final compiled program (all predictor stages) actually used for the reported test score. Optimization-run summaries (`best_program_as_per_agg_score`, `best_valset_agg_score`) and `output.log` corroborate which candidate was selected; the `new_instruction` summary key on the opt runs was `"Merged program"` (last iteration marker), not the final prompt itself.


---


## LiveBench-Math — GEPA

- Optimization run: `n7si0xa5` — /home/jovyan/gepa-artifact/experiment_runs_data/experiment_runs/seed_0/LiveBenchMathBench_CoT_GEPA_gpt-41-mini-openrouter (state: finished)
- Final-eval run: `8hllkzv8` — LiveBenchMathBench_CoT_GEPA_gpt-41-mini-openrouter_seed_0_final_eval (state: finished)
- Test score (final_eval `test/score`): **64.21**
- Best valset agg score (opt run): 76.96 (best_program_as_per_agg_score idx = 11)
- Number of predictor stages: 1
- Prompt source: final_eval `prompts/table` artifact


### Stage 0: `predict`
- Signature: `StringSignature(question -> reasoning, answer
    instructions="You will be given a mathematical problem stated as a question. Your objective is to produce a fully worked-out solution that is clear, thorough, and rigorously justified, showing every step necessary for complete transparency and verification.\n\nFollow these detailed guidelines precisely:\n\n1. **Problem Interpretation and Setup**  \n   - Restate the problem clearly in your own words and note any assumptions or conditions implied.  \n   - Identify known formulas, definitions, or theorems that are relevant and explicitly state them before use.  \n   - For problems involving parameters, variables, or geometric configurations, explain the meaning and relationships of all parameters clearly, including any natural or imposed constraints.\n\n2. **Step-by-step Detailed Reasoning**  \n   - Write out each algebraic or arithmetic step explicitly, including expansions, substitutions, factorizations, identity applications, rearrangements, and simplifications.  \n   - Use consistent and proper mathematical notation (e.g., LaTeX-style displayed equations).  \n   - For numeric calculations, retain exact forms throughout (fractions, roots, powers), and simplify radicals and fractions fully before resorting to decimals. Only use approximations if explicitly requested, and always provide exact symbolic forms alongside.  \n   - When dealing with polynomials, number theory, or roots of unity, invoke known properties, factorization patterns, or identities to justify simplifications.  \n   - For geometry problems involving conic sections, vectors, or distance and orthogonality constraints, employ appropriate parametrizations (e.g., hyperbolic functions for hyperbola points) and translate constraints into equations systematically. Define vectors and points clearly, and show how conditions translate to algebraic relations.  \n   - When maximizing or minimizing values under given constraints, provide detailed reasoning of dependency and extremum determination, verifying with derivative tests or inequality arguments if needed.  \n   - If substitutions or reparametrization are used, explain the reasoning behind these choices and their effect on the problem.  \n   - Explicitly verify that all steps maintain consistency with problem constraints (e.g., positivity, distinctness, integer values).  \n   \n3. **Answer Formatting and Presentation**  \n   - If asked to provide the final answer boxed, use \\(\\boxed{...}\\) notation for the final solution.  \n   - For multiple-choice problems where you must submit a repeated letter string corresponding to the answer, provide it exactly as requested with no additional text or formatting.  \n   - When formula matching or missing tag completion tasks are given, produce a strictly comma-separated sequence of formula identifiers corresponding to the missing tags, with no added commentary.  \n   - For numerical answers requiring exact integers or digit strings with possible leading zeros, output exactly as specified, with no trailing or extra text.  \n   - Avoid approximate or decimal answers unless the problem explicitly accepts or requests them.\n\n4. **Domain-Specific & Factual Details from Past Examples**  \n   - For systems of equations involving positive real variables, carefully analyze all cases including absolute values and consider domain restrictions explicitly.  \n   - When tackling tetrahedron problems with distances and edge lengths given, use coordinate geometry to locate points. Start by fixing a vertex and aligning one edge along an axis to reduce dimensional variables. Apply distance formulas explicitly and solve for coordinates component-wise.  \n   - Compute volumes using vector scalar triple products with explicit determinant calculations. Express volume in exact radical form and simplify.  \n   - Compute surface areas via Heron's formula for triangles, maintaining exact radical expressions until the final step.  \n   - Calculate the inradius (distance from incenter to faces) as \\( r = \\frac{3V}{S} \\), where \\(V\\) is volume and \\(S\\) is surface area, showing explicit substitution and simplification.  \n   - For sets or sequences of positive integers with constraints on sums, mode uniqueness, and median properties:  \n     * Analyze median implications on list length (odd vs even) and implications on presence or absence of median value in the list.  \n     * Consider parity and integer averaging for medians in even-length lists.  \n     * Use ordering conditions carefully to place mode values and ensure uniqueness by comparing frequencies.  \n     * Reason out feasible list lengths and attempt plausible constructions, verifying sum and uniqueness constraints fully.  \n     * When enumerating or guessing integers, carefully check sum, median, mode conditions, and uniqueness.  \n     * When finalizing answers (e.g., sum of squares), calculate with exact values, assembling all terms explicitly.\n\n5. **Verification and Completeness**  \n   - Recheck every algebraic or arithmetic step for accuracy and consistency with problem conditions.  \n   - Confirm that the final solution fully addresses the question asked (e.g., compute \\(x+y\\), inradius, sum of squares) in the required exact form and formatting.  \n   - In problems where inputs or hints suggest multiple cases, analyze all plausible cases and justify discarding invalid ones based on given conditions.  \n   - Keep all reasoning transparent enough so a reviewer or automated system can verify every step without guessing underlying assumptions.\n\n6. **General Style and Clarity**  \n   - Separate parts logically: problem restatement, assumptions, step-by-step solution, final conclusion.  \n   - Use consistent notation and clear explanations for each step — do not skip intermediate reasoning.  \n   - When referencing known mathematical facts, state them explicitly before applying.  \n   - Use exact symbolic forms throughout except when an explicit numeric approximation is requested or unavoidable.  \n   - Boldly mark final answers exactly as specified.\n\nFollowing this instruction will ensure your responses are precise, transparent, complete, and easily verifiable according to the highest standards of mathematical problem solving."
    question = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    answer = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Answer:', 'desc': '${answer}'})
)`
- Fields: `{"question": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "answer": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Answer:', 'desc': '${answer}'}"}`
- Instruction length: 6168 chars

Instruction (verbatim):

```text
You will be given a mathematical problem stated as a question. Your objective is to produce a fully worked-out solution that is clear, thorough, and rigorously justified, showing every step necessary for complete transparency and verification.

Follow these detailed guidelines precisely:

1. **Problem Interpretation and Setup**  
   - Restate the problem clearly in your own words and note any assumptions or conditions implied.  
   - Identify known formulas, definitions, or theorems that are relevant and explicitly state them before use.  
   - For problems involving parameters, variables, or geometric configurations, explain the meaning and relationships of all parameters clearly, including any natural or imposed constraints.

2. **Step-by-step Detailed Reasoning**  
   - Write out each algebraic or arithmetic step explicitly, including expansions, substitutions, factorizations, identity applications, rearrangements, and simplifications.  
   - Use consistent and proper mathematical notation (e.g., LaTeX-style displayed equations).  
   - For numeric calculations, retain exact forms throughout (fractions, roots, powers), and simplify radicals and fractions fully before resorting to decimals. Only use approximations if explicitly requested, and always provide exact symbolic forms alongside.  
   - When dealing with polynomials, number theory, or roots of unity, invoke known properties, factorization patterns, or identities to justify simplifications.  
   - For geometry problems involving conic sections, vectors, or distance and orthogonality constraints, employ appropriate parametrizations (e.g., hyperbolic functions for hyperbola points) and translate constraints into equations systematically. Define vectors and points clearly, and show how conditions translate to algebraic relations.  
   - When maximizing or minimizing values under given constraints, provide detailed reasoning of dependency and extremum determination, verifying with derivative tests or inequality arguments if needed.  
   - If substitutions or reparametrization are used, explain the reasoning behind these choices and their effect on the problem.  
   - Explicitly verify that all steps maintain consistency with problem constraints (e.g., positivity, distinctness, integer values).  
   
3. **Answer Formatting and Presentation**  
   - If asked to provide the final answer boxed, use \(\boxed{...}\) notation for the final solution.  
   - For multiple-choice problems where you must submit a repeated letter string corresponding to the answer, provide it exactly as requested with no additional text or formatting.  
   - When formula matching or missing tag completion tasks are given, produce a strictly comma-separated sequence of formula identifiers corresponding to the missing tags, with no added commentary.  
   - For numerical answers requiring exact integers or digit strings with possible leading zeros, output exactly as specified, with no trailing or extra text.  
   - Avoid approximate or decimal answers unless the problem explicitly accepts or requests them.

4. **Domain-Specific & Factual Details from Past Examples**  
   - For systems of equations involving positive real variables, carefully analyze all cases including absolute values and consider domain restrictions explicitly.  
   - When tackling tetrahedron problems with distances and edge lengths given, use coordinate geometry to locate points. Start by fixing a vertex and aligning one edge along an axis to reduce dimensional variables. Apply distance formulas explicitly and solve for coordinates component-wise.  
   - Compute volumes using vector scalar triple products with explicit determinant calculations. Express volume in exact radical form and simplify.  
   - Compute surface areas via Heron's formula for triangles, maintaining exact radical expressions until the final step.  
   - Calculate the inradius (distance from incenter to faces) as \( r = \frac{3V}{S} \), where \(V\) is volume and \(S\) is surface area, showing explicit substitution and simplification.  
   - For sets or sequences of positive integers with constraints on sums, mode uniqueness, and median properties:  
     * Analyze median implications on list length (odd vs even) and implications on presence or absence of median value in the list.  
     * Consider parity and integer averaging for medians in even-length lists.  
     * Use ordering conditions carefully to place mode values and ensure uniqueness by comparing frequencies.  
     * Reason out feasible list lengths and attempt plausible constructions, verifying sum and uniqueness constraints fully.  
     * When enumerating or guessing integers, carefully check sum, median, mode conditions, and uniqueness.  
     * When finalizing answers (e.g., sum of squares), calculate with exact values, assembling all terms explicitly.

5. **Verification and Completeness**  
   - Recheck every algebraic or arithmetic step for accuracy and consistency with problem conditions.  
   - Confirm that the final solution fully addresses the question asked (e.g., compute \(x+y\), inradius, sum of squares) in the required exact form and formatting.  
   - In problems where inputs or hints suggest multiple cases, analyze all plausible cases and justify discarding invalid ones based on given conditions.  
   - Keep all reasoning transparent enough so a reviewer or automated system can verify every step without guessing underlying assumptions.

6. **General Style and Clarity**  
   - Separate parts logically: problem restatement, assumptions, step-by-step solution, final conclusion.  
   - Use consistent notation and clear explanations for each step — do not skip intermediate reasoning.  
   - When referencing known mathematical facts, state them explicitly before applying.  
   - Use exact symbolic forms throughout except when an explicit numeric approximation is requested or unavoidable.  
   - Boldly mark final answers exactly as specified.

Following this instruction will ensure your responses are precise, transparent, complete, and easily verifiable according to the highest standards of mathematical problem solving.
```

---


## LiveBench-Math — GEPA-MERGE

- Optimization run: `omlg6qom` — /home/jovyan/gepa-artifact/experiment_runs_data/experiment_runs/seed_0/LiveBenchMathBench_CoT_GEPA-MERGE_gpt-41-mini-openrouter (state: finished)
- Final-eval run: `0rm9nuoc` — LiveBenchMathBench_CoT_GEPA-MERGE_gpt-41-mini-openrouter_seed_0_final_eval (state: finished)
- Test score (final_eval `test/score`): **61.14**
- Best valset agg score (opt run): 74.27 (best_program_as_per_agg_score idx = 6)
- Number of predictor stages: 1
- Prompt source: final_eval `prompts/table` artifact


### Stage 0: `predict`
- Signature: `StringSignature(question -> reasoning, answer
    instructions='You will be given a mathematical or quantitative problem to solve. Your task is to:\n\n1. Carefully read and understand the problem.\n2. Solve the question step-by-step, showing detailed and clear reasoning with intermediate calculations.\n3. Ensure exact arithmetic or algebraic manipulation is used when required, such as fractions or radicals, to provide an exact answer rather than a decimal approximation.\n4. Format and present the final answer according to the instructions in the problem prompt:\n   - If the prompt requests the answer in LaTeX format, enclose the final answer inside a `\\boxed{}` command (for example, `\\boxed{\\frac{2}{7}}`).\n   - If the prompt specifies a particular letter answer repetition format (e.g., repeating the letter five times), follow that exactly.\n5. Double-check your final answer both for arithmetic correctness and format correctness to ensure it meets the task’s expectations.\n6. When solving statistical problems like computing sample variance, recall that sample variance uses denominator \\(n - 1\\), where \\(n\\) is the sample size.\n7. For matrix operations, follow standard formulas precisely (e.g., determinant of 2x2 matrix \\( \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} \\) is \\(ad - bc\\)).\n8. When multiple-choice formats are given with lettered options, clearly identify the correct choice and output the answer in the exact way required by the prompt.\n\nThis instruction is intended to generate solutions that are both mathematically correct and formatted exactly as required for automated parsing and validation. Always revisit your calculations if your final answer conflicts with expected results, and adjust your work accordingly.'
    question = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    answer = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Answer:', 'desc': '${answer}'})
)`
- Fields: `{"question": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "answer": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Answer:', 'desc': '${answer}'}"}`
- Instruction length: 1679 chars

Instruction (verbatim):

```text
You will be given a mathematical or quantitative problem to solve. Your task is to:

1. Carefully read and understand the problem.
2. Solve the question step-by-step, showing detailed and clear reasoning with intermediate calculations.
3. Ensure exact arithmetic or algebraic manipulation is used when required, such as fractions or radicals, to provide an exact answer rather than a decimal approximation.
4. Format and present the final answer according to the instructions in the problem prompt:
   - If the prompt requests the answer in LaTeX format, enclose the final answer inside a `\boxed{}` command (for example, `\boxed{\frac{2}{7}}`).
   - If the prompt specifies a particular letter answer repetition format (e.g., repeating the letter five times), follow that exactly.
5. Double-check your final answer both for arithmetic correctness and format correctness to ensure it meets the task’s expectations.
6. When solving statistical problems like computing sample variance, recall that sample variance uses denominator \(n - 1\), where \(n\) is the sample size.
7. For matrix operations, follow standard formulas precisely (e.g., determinant of 2x2 matrix \( \begin{pmatrix} a & b \\ c & d \end{pmatrix} \) is \(ad - bc\)).
8. When multiple-choice formats are given with lettered options, clearly identify the correct choice and output the answer in the exact way required by the prompt.

This instruction is intended to generate solutions that are both mathematically correct and formatted exactly as required for automated parsing and validation. Always revisit your calculations if your final answer conflicts with expected results, and adjust your work accordingly.
```

---


## HoVer — GEPA

- Optimization run: `r11de69j` — /home/jovyan/gepa-artifact/experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_gpt-41-mini-openrouter (state: finished)
- Final-eval run: `grkqqy50` — hoverBench_HoverMultiHop_GEPA_gpt-41-mini-openrouter_seed_0_final_eval (state: finished)
- Test score (final_eval `test/score`): **50.33**
- Best valset agg score (opt run): 54.0 (best_program_as_per_agg_score idx = 20)
- Number of predictor stages: 4
- Prompt source: final_eval `prompts/table` artifact


### Stage 0: `create_query_hop2.predict`
- Signature: `StringSignature(claim, summary_1 -> reasoning, query
    instructions='Task Description:  \nYou are given two text fields as input: `claim` and `summary_1`. Your goal is to generate a `query` designed to retrieve evidence that can confirm, refute, or verify the given `claim` based on the information present or lacking in `summary_1`. This is typically a natural language question or set of questions that can be used as search queries to find relevant supporting or negating evidence for all parts of the claim. \n\nKey Points and Detailed Instructions:  \n1. The `claim` may consist of multiple factual assertions involving named entities (people, places, events), dates, titles (e.g., films, albums, plays), or roles (e.g., director, songwriter). Your queries must address all significant details in the claim.  \n\n2. The provided `summary_1` is a condensed background or related information that partially confirms or denies aspects of the claim or may leave some claim aspects unverified. Use it to identify which parts of the claim have supporting evidence and which parts lack evidence or require further verification.  \n\n3. Your `query` must be rich and comprehensive enough to retrieve all relevant evidence that can verify or disprove *each element* of the claim, especially those not already confirmed by `summary_1`. This means explicitly including all key entities or terms mentioned in the claim that are missing or ambiguous in `summary_1`.  \n\n4. Effectively connect all critical claim components in your query. For example, if the claim involves multiple entities or relationships (such as "A and B", or "person X and event Y"), ensure that your query strategically links these to find evidence linking the elements together.  \n\n5. When constructing queries:  \n   - Include all proper names, roles, titles, dates, locations, and any other distinctive identifiers from the claim, especially those absent or only partially present in `summary_1`.  \n   - Formulate your queries as clear and specific natural language questions or search prompts that directly address claim verification (e.g., "Who directed...", "Was ... filmed in...", "Did ... appear in ...").  \n   - Avoid vague or overly broad questions that might fail to retrieve precise evidence relevant to complex or multi-part claims.  \n\n6. Aim for queries that can discover new or missing evidence mentioned in claim but not yet covered in the summary. This means explicitly referencing people, titles, or facts highlighted as “missing evidence” in prior feedback.  \n\n7. The reasoning step (optional) can be used internally to identify which parts of the claim are confirmed or unverified by `summary_1` and guide your query construction, but the final output required is only the `query`.  \n\nExample Strategy Applied to Provided Examples:  \n- In Example 1, the assistant’s query missed including “Jay T. Wright,” who is crucial to differentiate from the director Lloyd Kaufman. A better query would mention both individuals explicitly to retrieve evidence about their distinct roles.  \n- In Example 3, the assistant forgot to mention relevant keywords like "Adam Guettel," "Elena Shaddow," or the song/play names like "How Glory Goes". Including these increases the chance of finding evidentiary documents.  \n- In Example 2, the assistant successfully linked the specific location with the musical film and director, showing a correct approach to forming queries for multi-part claims.  \n\nSummary:  \nTo produce effective queries, ensure they comprehensively incorporate all claim details, explicitly reference missing or ambiguous evidence mentioned or omitted in `summary_1`, and formulate clear, specific natural language questions to retrieve targeted supporting or refuting evidence for the claim.'
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    summary_1 = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    query = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "summary_1": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "query": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'}"}`
- Instruction length: 3700 chars

Instruction (verbatim):

```text
Task Description:  
You are given two text fields as input: `claim` and `summary_1`. Your goal is to generate a `query` designed to retrieve evidence that can confirm, refute, or verify the given `claim` based on the information present or lacking in `summary_1`. This is typically a natural language question or set of questions that can be used as search queries to find relevant supporting or negating evidence for all parts of the claim. 

Key Points and Detailed Instructions:  
1. The `claim` may consist of multiple factual assertions involving named entities (people, places, events), dates, titles (e.g., films, albums, plays), or roles (e.g., director, songwriter). Your queries must address all significant details in the claim.  

2. The provided `summary_1` is a condensed background or related information that partially confirms or denies aspects of the claim or may leave some claim aspects unverified. Use it to identify which parts of the claim have supporting evidence and which parts lack evidence or require further verification.  

3. Your `query` must be rich and comprehensive enough to retrieve all relevant evidence that can verify or disprove *each element* of the claim, especially those not already confirmed by `summary_1`. This means explicitly including all key entities or terms mentioned in the claim that are missing or ambiguous in `summary_1`.  

4. Effectively connect all critical claim components in your query. For example, if the claim involves multiple entities or relationships (such as "A and B", or "person X and event Y"), ensure that your query strategically links these to find evidence linking the elements together.  

5. When constructing queries:  
   - Include all proper names, roles, titles, dates, locations, and any other distinctive identifiers from the claim, especially those absent or only partially present in `summary_1`.  
   - Formulate your queries as clear and specific natural language questions or search prompts that directly address claim verification (e.g., "Who directed...", "Was ... filmed in...", "Did ... appear in ...").  
   - Avoid vague or overly broad questions that might fail to retrieve precise evidence relevant to complex or multi-part claims.  

6. Aim for queries that can discover new or missing evidence mentioned in claim but not yet covered in the summary. This means explicitly referencing people, titles, or facts highlighted as “missing evidence” in prior feedback.  

7. The reasoning step (optional) can be used internally to identify which parts of the claim are confirmed or unverified by `summary_1` and guide your query construction, but the final output required is only the `query`.  

Example Strategy Applied to Provided Examples:  
- In Example 1, the assistant’s query missed including “Jay T. Wright,” who is crucial to differentiate from the director Lloyd Kaufman. A better query would mention both individuals explicitly to retrieve evidence about their distinct roles.  
- In Example 3, the assistant forgot to mention relevant keywords like "Adam Guettel," "Elena Shaddow," or the song/play names like "How Glory Goes". Including these increases the chance of finding evidentiary documents.  
- In Example 2, the assistant successfully linked the specific location with the musical film and director, showing a correct approach to forming queries for multi-part claims.  

Summary:  
To produce effective queries, ensure they comprehensively incorporate all claim details, explicitly reference missing or ambiguous evidence mentioned or omitted in `summary_1`, and formulate clear, specific natural language questions to retrieve targeted supporting or refuting evidence for the claim.
```

### Stage 1: `create_query_hop3.predict`
- Signature: `StringSignature(claim, summary_1, summary_2 -> reasoning, query
    instructions='Task Description:\nYou will be provided with three fields as input: `claim`, `summary_1`, and `summary_2`. Your goal is to generate a `query` that is most useful for retrieving or verifying evidence directly relevant to the claim, considering the information presented in both summaries.\n\nInput Format:\n- `claim`: A factual or composite statement that requires verification or fact-checking.\n- `summary_1`: A concise summary of evidence or factual information related to the claim.\n- `summary_2`: Another concise but potentially complementary or expanded summary of evidence related to the claim.\n\nOutput Format:\n- `query`: A carefully constructed natural language question or search query designed to help retrieve the most pertinent evidence to confirm, refute, or clarify the claim based on the summaries provided.\n\nDetailed Instructions:\n\n1. Understand the Claim and Evidence:\n   - Fully comprehend the factual components and logical structure of the claim.\n   - Analyze both summaries for information that supports, refutes, or partially supports the claim.\n\n2. Identify Evidence Gaps and Connections:\n   - Note any contradictions or nuances between the claim and the summaries.\n   - Identify key entities (people, works, events), dates, attributes, or relationships mentioned in the summaries or claim that are crucial to establishing the claim’s truth.\n   - Pay special attention to domain-specific details such as roles (e.g., poet, novelist), historical events, professional titles, artistic works, competitions, and their attributes (e.g., number of acts).\n\n3. Constructing the Query:\n   - Frame the query to explicitly target the precise piece(s) of missing, ambiguous, or pivotal evidence required to verify the claim.\n   - Where possible, incorporate domain-specific terminology or named entities from the summaries and claim to improve query relevance.\n   - Formulate the query as a clear, focused natural language question or search phrase that could be used to retrieve key evidence.\n   - Anticipate the type of evidence needed (e.g., counts, professional associations, event participation) and guide the query accordingly.\n\n4. Addressing Missing or Overlooked Evidence:\n   - Ensure that important entities or facts mentioned in the summaries but not explicitly tied to the claim in earlier queries are included, to broaden evidence retrieval.\n   - For claims with multiple components or comparisons, consider queries that explore these relationships explicitly.\n\n5. Examples of Query Improvements:\n   - If a domain-relevant person, event, or role is mentioned in the summaries but not linked to the claim in the initial query, include them explicitly.\n   - For comparative claims (e.g., number of acts in operas), queries should ask directly about that comparison.\n   - For claims involving partnerships or associations (e.g., sport doubles partners), include both parties’ names and event details in the query.\n   - For verifying roles or titles held by individuals, query the current official positions and confirm multiple roles if relevant.\n\n6. Deliver one query per input, phrased clearly and succinctly, that when searched would yield the strongest evidence to support or refute the claim based on the given summaries.\n\nSummary:\nYour output should be a natural language query optimized to uncover the most relevant missing or unclear evidence needed for claim validation, using domain-specific details and ensuring that any overlooked or nuanced information in the summaries is considered and included in the query formulation.'
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    summary_1 = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'})
    summary_2 = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 2:', 'desc': '${summary_2}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    query = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "summary_1": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'}", "summary_2": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 2:', 'desc': '${summary_2}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "query": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'}"}`
- Instruction length: 3549 chars

Instruction (verbatim):

```text
Task Description:
You will be provided with three fields as input: `claim`, `summary_1`, and `summary_2`. Your goal is to generate a `query` that is most useful for retrieving or verifying evidence directly relevant to the claim, considering the information presented in both summaries.

Input Format:
- `claim`: A factual or composite statement that requires verification or fact-checking.
- `summary_1`: A concise summary of evidence or factual information related to the claim.
- `summary_2`: Another concise but potentially complementary or expanded summary of evidence related to the claim.

Output Format:
- `query`: A carefully constructed natural language question or search query designed to help retrieve the most pertinent evidence to confirm, refute, or clarify the claim based on the summaries provided.

Detailed Instructions:

1. Understand the Claim and Evidence:
   - Fully comprehend the factual components and logical structure of the claim.
   - Analyze both summaries for information that supports, refutes, or partially supports the claim.

2. Identify Evidence Gaps and Connections:
   - Note any contradictions or nuances between the claim and the summaries.
   - Identify key entities (people, works, events), dates, attributes, or relationships mentioned in the summaries or claim that are crucial to establishing the claim’s truth.
   - Pay special attention to domain-specific details such as roles (e.g., poet, novelist), historical events, professional titles, artistic works, competitions, and their attributes (e.g., number of acts).

3. Constructing the Query:
   - Frame the query to explicitly target the precise piece(s) of missing, ambiguous, or pivotal evidence required to verify the claim.
   - Where possible, incorporate domain-specific terminology or named entities from the summaries and claim to improve query relevance.
   - Formulate the query as a clear, focused natural language question or search phrase that could be used to retrieve key evidence.
   - Anticipate the type of evidence needed (e.g., counts, professional associations, event participation) and guide the query accordingly.

4. Addressing Missing or Overlooked Evidence:
   - Ensure that important entities or facts mentioned in the summaries but not explicitly tied to the claim in earlier queries are included, to broaden evidence retrieval.
   - For claims with multiple components or comparisons, consider queries that explore these relationships explicitly.

5. Examples of Query Improvements:
   - If a domain-relevant person, event, or role is mentioned in the summaries but not linked to the claim in the initial query, include them explicitly.
   - For comparative claims (e.g., number of acts in operas), queries should ask directly about that comparison.
   - For claims involving partnerships or associations (e.g., sport doubles partners), include both parties’ names and event details in the query.
   - For verifying roles or titles held by individuals, query the current official positions and confirm multiple roles if relevant.

6. Deliver one query per input, phrased clearly and succinctly, that when searched would yield the strongest evidence to support or refute the claim based on the given summaries.

Summary:
Your output should be a natural language query optimized to uncover the most relevant missing or unclear evidence needed for claim validation, using domain-specific details and ensuring that any overlooked or nuanced information in the summaries is considered and included in the query formulation.
```

### Stage 2: `summarize1.predict`
- Signature: `StringSignature(claim, passages -> reasoning, summary
    instructions="Task Description:\nYou are given a `claim` and a list of relevant `passages`. Your task is to produce a concise and factually accurate `summary` that directly addresses the truthfulness or validity of the claim based on the evidence available in the passages.\n\nInput Format:\n- `claim`: A single statement or claim to verify or summarize.\n- `passages`: A list of text passages containing information relevant to the claim.\n\nOutput Format:\n- `summary`: A brief, clear summary that either supports, refutes, or states the insufficiency of evidence regarding the claim, explicitly referring to relevant details found in the passages.\n\nDetailed Instructions:\n\n1. **Comprehension:** \n   - Understand the claim fully, including all key entities, dates, relationships, and qualifiers.\n   - Carefully read the passages to identify relevant evidence that supports, contradicts, or is insufficient to assess the claim.\n\n2. **Evidence Linking and Reasoning:**\n   - Cross-reference the entities, dates, and facts mentioned in the claim with those described in the passages.\n   - Highlight core connections such as birth dates, authorship, film roles, production credits, nationalities, brand histories, or other domain-specific details as appropriate.\n   - If the passages do not mention the claim’s main entities or lack relevant connections, state that evidence is missing or inconclusive.\n\n3. **Avoid Assumptions:**\n   - Avoid inferring or assuming facts that are not directly or indirectly supported by the passages.\n   - If some information (e.g., birthdate of an author) is missing but can be logically inferred (e.g., an author must be born before publishing a book), carefully note the reasoning behind the inference.\n\n4. **Domain-Specific Details to Leverage:**\n   - Composer and songwriter identities, their nationalities, and their contributions (e.g., commercials, film soundtracks).\n   - Author birthdates and publication years to place claims about chronology in context.\n   - Film cast, roles, film genres, production credits including directors, co-producers, and release years.\n   - Brands and companies’ histories and geographical presence for claims about their background.\n   - Specific references to well-known franchises, events, or works (e.g., Eurovision, books, films) and their creators or participants.\n\n5. **Summary Composition:**\n   - Clearly state whether the claim is supported, contradicted, or unverifiable based on the passages.\n   - Include key factual details from the passages that justify your conclusion.\n   - If correcting inaccuracies in the claim, explicitly mention what is correct and what is incorrect.\n   - Keep the summary succinct without losing critical nuance that affects the claim's truthfulness.\n\n6. **Usefulness for Query Generation:**\n   - Write the summary to facilitate future query generation by clearly referencing key entities or facts that serve as evidence or highlight missing information.\n   - Mention missing evidence explicitly if there is none in the passages, to guide further document retrieval.\n\nExample Summary Templates:\n- Supported: “[Entity] was born in [year], prior to [other entity or event], which supports the claim.”\n- Refuted: “[Entity] is not [nationality] but [correct nationality], contradicting the claim.”\n- Insufficient evidence: “The passages do not provide any information about [key aspect], so the claim cannot be verified.”\n\nBy following the above instructions, you will create accurate, precise, and evidence-based summaries that help validate or evaluate claims with clarity and domain specificity."
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    passages = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    summary = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "passages": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "summary": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'}"}`
- Instruction length: 3578 chars

Instruction (verbatim):

```text
Task Description:
You are given a `claim` and a list of relevant `passages`. Your task is to produce a concise and factually accurate `summary` that directly addresses the truthfulness or validity of the claim based on the evidence available in the passages.

Input Format:
- `claim`: A single statement or claim to verify or summarize.
- `passages`: A list of text passages containing information relevant to the claim.

Output Format:
- `summary`: A brief, clear summary that either supports, refutes, or states the insufficiency of evidence regarding the claim, explicitly referring to relevant details found in the passages.

Detailed Instructions:

1. **Comprehension:** 
   - Understand the claim fully, including all key entities, dates, relationships, and qualifiers.
   - Carefully read the passages to identify relevant evidence that supports, contradicts, or is insufficient to assess the claim.

2. **Evidence Linking and Reasoning:**
   - Cross-reference the entities, dates, and facts mentioned in the claim with those described in the passages.
   - Highlight core connections such as birth dates, authorship, film roles, production credits, nationalities, brand histories, or other domain-specific details as appropriate.
   - If the passages do not mention the claim’s main entities or lack relevant connections, state that evidence is missing or inconclusive.

3. **Avoid Assumptions:**
   - Avoid inferring or assuming facts that are not directly or indirectly supported by the passages.
   - If some information (e.g., birthdate of an author) is missing but can be logically inferred (e.g., an author must be born before publishing a book), carefully note the reasoning behind the inference.

4. **Domain-Specific Details to Leverage:**
   - Composer and songwriter identities, their nationalities, and their contributions (e.g., commercials, film soundtracks).
   - Author birthdates and publication years to place claims about chronology in context.
   - Film cast, roles, film genres, production credits including directors, co-producers, and release years.
   - Brands and companies’ histories and geographical presence for claims about their background.
   - Specific references to well-known franchises, events, or works (e.g., Eurovision, books, films) and their creators or participants.

5. **Summary Composition:**
   - Clearly state whether the claim is supported, contradicted, or unverifiable based on the passages.
   - Include key factual details from the passages that justify your conclusion.
   - If correcting inaccuracies in the claim, explicitly mention what is correct and what is incorrect.
   - Keep the summary succinct without losing critical nuance that affects the claim's truthfulness.

6. **Usefulness for Query Generation:**
   - Write the summary to facilitate future query generation by clearly referencing key entities or facts that serve as evidence or highlight missing information.
   - Mention missing evidence explicitly if there is none in the passages, to guide further document retrieval.

Example Summary Templates:
- Supported: “[Entity] was born in [year], prior to [other entity or event], which supports the claim.”
- Refuted: “[Entity] is not [nationality] but [correct nationality], contradicting the claim.”
- Insufficient evidence: “The passages do not provide any information about [key aspect], so the claim cannot be verified.”

By following the above instructions, you will create accurate, precise, and evidence-based summaries that help validate or evaluate claims with clarity and domain specificity.
```

### Stage 3: `summarize2.predict`
- Signature: `StringSignature(claim, context, passages -> reasoning, summary
    instructions='Task Description:\n\nYou will be provided with three fields: `claim`, `context`, and `passages`. Your goal is to produce a concise and informative `summary` that synthesizes the information from the `context` and the `passages` in relation to the `claim`.\n\nDetailed Instructions:\n\n1. Understand the Claim:\n   - Carefully read the `claim` to identify the key entities, facts, relationships, or events being asserted.\n   - Identify any implicit or explicit comparisons, temporal relations, or attributions made in the claim.\n\n2. Analyze the Context:\n   - Review the provided `context` which often includes background information or an assessment of the claim’s truthfulness.\n   - Use the `context` as the starting point for reasoning but verify with the evidence found in the `passages`.\n\n3. Examine the Passages:\n   - The `passages` contain relevant or potentially relevant information, such as descriptions, biographies, filmographies, product details, or historical facts.\n   - Identify and extract any direct evidence (dates, roles, relationships, flavors, productions, etc.) that support, contradict, or clarify parts of the claim.\n   - Pay special attention to key named entities or terms in the claim and explicitly connect them to matching evidence in the passages (e.g., linking a film director’s name found in the passages to the claim’s referenced film).\n\n4. Reasoning and Connecting Evidence:\n   - If a direct piece of evidence in the passages matches a key component of the claim, highlight it in your reasoning.\n   - If certain critical pieces of evidence are missing (e.g., a birthdate or a flavor specification), explicitly state that the claim cannot be fully confirmed or refuted based on the information provided.\n   - Avoid introducing information not found explicitly in the `passages` or `context`.\n   - Where multiple related entities or terms appear (e.g., different characters played by the same actor), clarify these relationships to avoid conflation.\n\n5. Produce the Summary:\n   - The summary should briefly state the claim’s status according to the provided information: confirmed, refuted, or unverified.\n   - Include specific references to the key evidence from the passages that underpin your conclusion.\n   - Ensure the summary facilitates the generation of precise and relevant search queries by explicitly mentioning relevant entities and evidence keywords.\n   - Avoid vague language; be precise about what is and is not supported by the evidence.\n   - The summary must reflect the reasoning step and mention explicit links between claim components and evidential passages.\n\n6. Additional Best Practices:\n   - When the claim involves comparing individuals, dates, or attributes, mention both parties and any relevant dates or roles.\n   - When the claim cites media (films, shows), chemistry (flavors), or historical data, clearly identify the related item in the passages.\n   - If the passages do not contain information about a critical element (such as a birthdate, flavor, or direct link), explicitly note that the claim is unverifiable with the given information.\n   - Name key entities and attributes relevant to the claim plainly in the summary to help retrieval systems locate precise evidence.\n\nSummary of Requirements:\n- Base your summary solely on the provided `context` and `passages`.\n- Highlight key evidence from the passages connecting explicitly to the claim.\n- State clearly whether the claim is supported, refuted, or unverifiable given the data.\n- Help downstream evidence retrieval by explicitly naming critical entities or keywords.\n\nExample structure for your summary:\n\n"[Entity/Person/Item] is [a confirmed or refuted attribute or role] according to the passages, which state [specific evidence]. However, [mention missing evidence or inability to fully confirm/refute claim] because [reason]."\n\nThis approach will ensure high-quality, evidence-focused summaries that facilitate evidence retrieval and claim verification.'
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    context = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Context:', 'desc': '${context}'})
    passages = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    summary = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "context": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Context:', 'desc': '${context}'}", "passages": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "summary": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'}"}`
- Instruction length: 3974 chars

Instruction (verbatim):

```text
Task Description:

You will be provided with three fields: `claim`, `context`, and `passages`. Your goal is to produce a concise and informative `summary` that synthesizes the information from the `context` and the `passages` in relation to the `claim`.

Detailed Instructions:

1. Understand the Claim:
   - Carefully read the `claim` to identify the key entities, facts, relationships, or events being asserted.
   - Identify any implicit or explicit comparisons, temporal relations, or attributions made in the claim.

2. Analyze the Context:
   - Review the provided `context` which often includes background information or an assessment of the claim’s truthfulness.
   - Use the `context` as the starting point for reasoning but verify with the evidence found in the `passages`.

3. Examine the Passages:
   - The `passages` contain relevant or potentially relevant information, such as descriptions, biographies, filmographies, product details, or historical facts.
   - Identify and extract any direct evidence (dates, roles, relationships, flavors, productions, etc.) that support, contradict, or clarify parts of the claim.
   - Pay special attention to key named entities or terms in the claim and explicitly connect them to matching evidence in the passages (e.g., linking a film director’s name found in the passages to the claim’s referenced film).

4. Reasoning and Connecting Evidence:
   - If a direct piece of evidence in the passages matches a key component of the claim, highlight it in your reasoning.
   - If certain critical pieces of evidence are missing (e.g., a birthdate or a flavor specification), explicitly state that the claim cannot be fully confirmed or refuted based on the information provided.
   - Avoid introducing information not found explicitly in the `passages` or `context`.
   - Where multiple related entities or terms appear (e.g., different characters played by the same actor), clarify these relationships to avoid conflation.

5. Produce the Summary:
   - The summary should briefly state the claim’s status according to the provided information: confirmed, refuted, or unverified.
   - Include specific references to the key evidence from the passages that underpin your conclusion.
   - Ensure the summary facilitates the generation of precise and relevant search queries by explicitly mentioning relevant entities and evidence keywords.
   - Avoid vague language; be precise about what is and is not supported by the evidence.
   - The summary must reflect the reasoning step and mention explicit links between claim components and evidential passages.

6. Additional Best Practices:
   - When the claim involves comparing individuals, dates, or attributes, mention both parties and any relevant dates or roles.
   - When the claim cites media (films, shows), chemistry (flavors), or historical data, clearly identify the related item in the passages.
   - If the passages do not contain information about a critical element (such as a birthdate, flavor, or direct link), explicitly note that the claim is unverifiable with the given information.
   - Name key entities and attributes relevant to the claim plainly in the summary to help retrieval systems locate precise evidence.

Summary of Requirements:
- Base your summary solely on the provided `context` and `passages`.
- Highlight key evidence from the passages connecting explicitly to the claim.
- State clearly whether the claim is supported, refuted, or unverifiable given the data.
- Help downstream evidence retrieval by explicitly naming critical entities or keywords.

Example structure for your summary:

"[Entity/Person/Item] is [a confirmed or refuted attribute or role] according to the passages, which state [specific evidence]. However, [mention missing evidence or inability to fully confirm/refute claim] because [reason]."

This approach will ensure high-quality, evidence-focused summaries that facilitate evidence retrieval and claim verification.
```

---


## HoVer — GEPA-MERGE

- Optimization run: `lo5sbuxv` — /home/jovyan/gepa-artifact/experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA-MERGE_gpt-41-mini-openrouter (state: finished)
- Final-eval run: `lgwdl9nf` — hoverBench_HoverMultiHop_GEPA-MERGE_gpt-41-mini-openrouter_seed_0_final_eval (state: finished)
- Test score (final_eval `test/score`): **49.67**
- Best valset agg score (opt run): 52.67 (best_program_as_per_agg_score idx = 14)
- Number of predictor stages: 4
- Prompt source: final_eval `prompts/table` artifact


### Stage 0: `create_query_hop2.predict`
- Signature: `StringSignature(claim, summary_1 -> reasoning, query
    instructions='Task Description:\nYou will be given two input fields: `claim` and `summary_1`. Your goal is to produce a `query` that can be used to retrieve evidence supporting or refuting the claim based on the information in `summary_1`.\n\nDetailed Instructions:\n\n1. Understand the Claim and Summary:\n   - The `claim` is a factual statement or assertion that requires verification.\n   - The `summary_1` provides background information, partial verification, or contradictions related to the claim, but may not fully confirm or deny the claim.\n   - Your `query` should be designed to identify or retrieve additional evidence that directly supports or challenges the key elements of the claim.\n\n2. Constructing Effective Queries:\n   - Extract all critical entities, facts, and relationships from the `claim` that require validation.\n   - Use the `summary_1` to identify which parts of the claim are confirmed, contradicted, or missing evidence.\n   - Incorporate relevant keywords, names, dates, titles, or distinctive identifiers mentioned in the claim and summary.\n   - Make the connections explicit: If the claim references a relationship or event involving multiple entities or concepts, include these connections in the query.\n   - Queries should be specific and targeted enough to retrieve relevant documents or passages that can confirm or deny the claim.\n   - Avoid vague or overly broad queries that do not help distinguish between the truth or falsehood of the claim.\n\n3. Leveraging Domain-Specific Knowledge:\n   - Be aware of niche or domain-specific terminology (e.g., film character names, song titles, actor nicknames, biographical book titles).\n   - When relevant, include alternative names or related terms to link potentially related evidence (e.g., for a film inspired by a character, include the film’s name alongside the character’s name).\n   - Include connections to notable attributes or contextual data (e.g., sales figures related to albums, years connected to song covers, author names linked to biographies).\n\n4. Generalizable Strategy:\n   - Verify who or what is attributed to the central claim (e.g., who starred in a film, what album contained a song).\n   - Query relationships clearly to retrieve text bridging elements mentioned but not fully connected in the summary.\n   - Ensure the query focuses on the disputed or unsupported elements of the claim, especially those noted as missing in the feedback.\n   - Use structured, natural-language questions or keyword conjunctions to improve the likelihood of retrieving precise evidence.\n\nExample:\nIf the claim is about an actor starring in a film inspired by a character, and the summary discusses the character and film but denies the actor’s involvement, your query should explicitly link:\n- The character name\n- The film title\n- The actor\'s name and nickname (if relevant)\ne.g., "Did [Actor Name], known as [Nickname], star in the film [Film Title] inspired by [Character Name]?"\n\nOutcome:\nBy following these instructions, your generated queries will better target the critical evidence needed to assess the claim’s validity, addressing gaps noted in the feedback and ensuring all relevant entities and relationships are incorporated for precise verification.'
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    summary_1 = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    query = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "summary_1": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "query": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'}"}`
- Instruction length: 3223 chars

Instruction (verbatim):

```text
Task Description:
You will be given two input fields: `claim` and `summary_1`. Your goal is to produce a `query` that can be used to retrieve evidence supporting or refuting the claim based on the information in `summary_1`.

Detailed Instructions:

1. Understand the Claim and Summary:
   - The `claim` is a factual statement or assertion that requires verification.
   - The `summary_1` provides background information, partial verification, or contradictions related to the claim, but may not fully confirm or deny the claim.
   - Your `query` should be designed to identify or retrieve additional evidence that directly supports or challenges the key elements of the claim.

2. Constructing Effective Queries:
   - Extract all critical entities, facts, and relationships from the `claim` that require validation.
   - Use the `summary_1` to identify which parts of the claim are confirmed, contradicted, or missing evidence.
   - Incorporate relevant keywords, names, dates, titles, or distinctive identifiers mentioned in the claim and summary.
   - Make the connections explicit: If the claim references a relationship or event involving multiple entities or concepts, include these connections in the query.
   - Queries should be specific and targeted enough to retrieve relevant documents or passages that can confirm or deny the claim.
   - Avoid vague or overly broad queries that do not help distinguish between the truth or falsehood of the claim.

3. Leveraging Domain-Specific Knowledge:
   - Be aware of niche or domain-specific terminology (e.g., film character names, song titles, actor nicknames, biographical book titles).
   - When relevant, include alternative names or related terms to link potentially related evidence (e.g., for a film inspired by a character, include the film’s name alongside the character’s name).
   - Include connections to notable attributes or contextual data (e.g., sales figures related to albums, years connected to song covers, author names linked to biographies).

4. Generalizable Strategy:
   - Verify who or what is attributed to the central claim (e.g., who starred in a film, what album contained a song).
   - Query relationships clearly to retrieve text bridging elements mentioned but not fully connected in the summary.
   - Ensure the query focuses on the disputed or unsupported elements of the claim, especially those noted as missing in the feedback.
   - Use structured, natural-language questions or keyword conjunctions to improve the likelihood of retrieving precise evidence.

Example:
If the claim is about an actor starring in a film inspired by a character, and the summary discusses the character and film but denies the actor’s involvement, your query should explicitly link:
- The character name
- The film title
- The actor's name and nickname (if relevant)
e.g., "Did [Actor Name], known as [Nickname], star in the film [Film Title] inspired by [Character Name]?"

Outcome:
By following these instructions, your generated queries will better target the critical evidence needed to assess the claim’s validity, addressing gaps noted in the feedback and ensuring all relevant entities and relationships are incorporated for precise verification.
```

### Stage 1: `create_query_hop3.predict`
- Signature: `StringSignature(claim, summary_1, summary_2 -> reasoning, query
    instructions='Task Description:  \nYou will be given three text fields as input: `claim`, `summary_1`, and `summary_2`. Your goal is to generate a `query` — a carefully crafted, clear, and concise question or set of questions designed to help retrieve evidence that directly assesses the truthfulness of the claim by making relevant connections between the claim and the information in the summaries.\n\nInput Format:  \n- `claim`: a statement or assertion that requires verification. It may contain multiple components, entities, relationships, or events.   \n- `summary_1` and `summary_2`: short, synthesized texts that contain information pertinent to verifying the claim. These summaries may partially support, refute, or remain neutral about the claim.\n\nExpected Output:  \n- `query`: a composed query (usually in interrogative form) intended for evidence retrieval systems to find documents or facts that help to confirm or refute the claim.\n\nDetailed Instructions:  \n1. **Understand the claim fully**: Identify all key entities, roles, relationships, dates, and attributes included in the claim.  \n2. **Analyze the summaries**: Determine what information they contain that supports, refutes, or leaves uncertain aspects of the claim, including any proper nouns, titles, names, dates, or relationships present.  \n3. **Incorporate critical keywords from both the claim and summaries into the query**: To maximize retrieval of relevant evidence, your query must explicitly mention all critical components and connecting terms found in both the claim and the summaries. For example, if the claim involves a film, an actor, a role, and a director, include all these elements by name.  \n4. **Formulate the query as a direct, clear question or compound question** that logically links these entities and relationships—this helps evidence retrievers locate information distinguishing between verified and unverified aspects.  \n5. **Use entity disambiguation in the query if needed**: For example, clarify movie titles, actor names, roles, dates, or affiliations to increase precision.  \n6. **Avoid vague or overly broad queries** that omit crucial named entities or relationships, as these limit retrieval effectiveness.  \n7. **If multiple assertions exist within the claim, the query should attempt to cover them all succinctly** but clearly.  \n\nAdditional Domain-Specific Notes:  \n- The claims often involve verifying casting or role relationships in films, verifying associations between people and institutions, or confirming production credits.  \n- Key named entities typically include actor names, movie titles, character names, director names, organizational affiliations, and dates.  \n- Summaries may confirm some relationships but often fail to resolve the full claim; your query should seek to connect the missing links (e.g., confirming whether a certain actor is the star of a film or whether a specific director directed a certain person).  \n- Queries should incorporate synonymous or related terms mentioned in summaries to expand the evidence retrieval scope where appropriate.  \n- Use well-structured, natural-language question format rather than keyword lists to improve retrieval accuracy.\n\nBy following these guidelines, your generated query will efficiently assist in retrieving evidence that covers all necessary facts to verify the claim accurately.'
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    summary_1 = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'})
    summary_2 = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 2:', 'desc': '${summary_2}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    query = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "summary_1": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 1:', 'desc': '${summary_1}'}", "summary_2": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Summary 2:', 'desc': '${summary_2}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "query": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Query:', 'desc': '${query}'}"}`
- Instruction length: 3350 chars

Instruction (verbatim):

```text
Task Description:  
You will be given three text fields as input: `claim`, `summary_1`, and `summary_2`. Your goal is to generate a `query` — a carefully crafted, clear, and concise question or set of questions designed to help retrieve evidence that directly assesses the truthfulness of the claim by making relevant connections between the claim and the information in the summaries.

Input Format:  
- `claim`: a statement or assertion that requires verification. It may contain multiple components, entities, relationships, or events.   
- `summary_1` and `summary_2`: short, synthesized texts that contain information pertinent to verifying the claim. These summaries may partially support, refute, or remain neutral about the claim.

Expected Output:  
- `query`: a composed query (usually in interrogative form) intended for evidence retrieval systems to find documents or facts that help to confirm or refute the claim.

Detailed Instructions:  
1. **Understand the claim fully**: Identify all key entities, roles, relationships, dates, and attributes included in the claim.  
2. **Analyze the summaries**: Determine what information they contain that supports, refutes, or leaves uncertain aspects of the claim, including any proper nouns, titles, names, dates, or relationships present.  
3. **Incorporate critical keywords from both the claim and summaries into the query**: To maximize retrieval of relevant evidence, your query must explicitly mention all critical components and connecting terms found in both the claim and the summaries. For example, if the claim involves a film, an actor, a role, and a director, include all these elements by name.  
4. **Formulate the query as a direct, clear question or compound question** that logically links these entities and relationships—this helps evidence retrievers locate information distinguishing between verified and unverified aspects.  
5. **Use entity disambiguation in the query if needed**: For example, clarify movie titles, actor names, roles, dates, or affiliations to increase precision.  
6. **Avoid vague or overly broad queries** that omit crucial named entities or relationships, as these limit retrieval effectiveness.  
7. **If multiple assertions exist within the claim, the query should attempt to cover them all succinctly** but clearly.  

Additional Domain-Specific Notes:  
- The claims often involve verifying casting or role relationships in films, verifying associations between people and institutions, or confirming production credits.  
- Key named entities typically include actor names, movie titles, character names, director names, organizational affiliations, and dates.  
- Summaries may confirm some relationships but often fail to resolve the full claim; your query should seek to connect the missing links (e.g., confirming whether a certain actor is the star of a film or whether a specific director directed a certain person).  
- Queries should incorporate synonymous or related terms mentioned in summaries to expand the evidence retrieval scope where appropriate.  
- Use well-structured, natural-language question format rather than keyword lists to improve retrieval accuracy.

By following these guidelines, your generated query will efficiently assist in retrieving evidence that covers all necessary facts to verify the claim accurately.
```

### Stage 2: `summarize1.predict`
- Signature: `StringSignature(claim, passages -> reasoning, summary
    instructions='Task Description:\nGiven structured inputs containing two fields, `claim` and `passages`, your goal is to produce a concise, factual, and evidence-based `summary` that directly addresses the truthfulness or accuracy of the claim using the information explicitly or implicitly present in the passages. The `summary` should synthesize relevant evidence to confirm, refute, or partially qualify the claim, highlighting supporting or contradictory facts found in the passages.\n\nInput Format:\n- `claim`: A statement or assertion, often complex, sometimes containing multiple parts, which may involve named entities, numbers, or factual relationships.\n- `passages`: A list of textual passages, each potentially containing biographical, historical, statistical, or contextual information related to the claim or its entities. Passages can include descriptions about people, organizations, locations, events, or works and may provide implicit or explicit evidence relevant to verifying the claim.\n\nOutput Format:\n- `summary`: A succinct paragraph that:\n  1. References key pieces of evidence from the provided passages.\n  2. Explicitly indicates whether the claim is fully true, false, or partially true.\n  3. Highlights any discrepancies, missing links, or clarifications found based on evidence.\n  4. Clearly distinguishes factual information from assumptions or unsupported conjectures.\n  5. Connects claim elements to relevant entities, titles, or numeric data mentioned in the passages, even if the connection is indirect and requires reasoning.\n\nImportant Domain-Specific Guidelines and Strategies:\n- Accurately identify the entities and specific details mentioned in the claim (e.g., authors, titles, locations, dates, numbers).\n- Disambiguate entities if multiple similar names or works appear in the passages (e.g., distinguishing a sci-fi author associated with "The Broken Tower" from others).\n- Use numeric or quantitative data critically; evaluate whether numbers in the claim are supported, exaggerated, or contradicted by evidence.\n- For claims referencing relationships (e.g., a person’s career influencing their work), highlight whether the passage explicitly confirms or negates this.\n- If no direct evidence connects a claim element (such as a title or name) to the stated fact, clearly state the absence of support, noting relevant related details in the passages.\n- When available data matches or partially matches the claim, accurately describe the level of support (e.g., “partially true,” “false in magnitude,” etc.).\n- Summaries should be crafted so they can effectively guide subsequent information retrieval or query generation for more evidence, hence linking claim terms and evidence clearly and explicitly is essential.\n- Avoid introducing external knowledge beyond what is logically inferable from the passages unless specifically stated.\n- When quantifying differences or comparisons (e.g., population sizes), critically analyze the plausibility and mention numeric relationships rather than absolute values when evidence is lacking.\n\nOverall, your output should demonstrate careful reasoning that draws out the relevant factual connections between the claim and the evidence passages, producing an informative yet focused summary useful for further verification or querying.'
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    passages = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    summary = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "passages": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "summary": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'}"}`
- Instruction length: 3298 chars

Instruction (verbatim):

```text
Task Description:
Given structured inputs containing two fields, `claim` and `passages`, your goal is to produce a concise, factual, and evidence-based `summary` that directly addresses the truthfulness or accuracy of the claim using the information explicitly or implicitly present in the passages. The `summary` should synthesize relevant evidence to confirm, refute, or partially qualify the claim, highlighting supporting or contradictory facts found in the passages.

Input Format:
- `claim`: A statement or assertion, often complex, sometimes containing multiple parts, which may involve named entities, numbers, or factual relationships.
- `passages`: A list of textual passages, each potentially containing biographical, historical, statistical, or contextual information related to the claim or its entities. Passages can include descriptions about people, organizations, locations, events, or works and may provide implicit or explicit evidence relevant to verifying the claim.

Output Format:
- `summary`: A succinct paragraph that:
  1. References key pieces of evidence from the provided passages.
  2. Explicitly indicates whether the claim is fully true, false, or partially true.
  3. Highlights any discrepancies, missing links, or clarifications found based on evidence.
  4. Clearly distinguishes factual information from assumptions or unsupported conjectures.
  5. Connects claim elements to relevant entities, titles, or numeric data mentioned in the passages, even if the connection is indirect and requires reasoning.

Important Domain-Specific Guidelines and Strategies:
- Accurately identify the entities and specific details mentioned in the claim (e.g., authors, titles, locations, dates, numbers).
- Disambiguate entities if multiple similar names or works appear in the passages (e.g., distinguishing a sci-fi author associated with "The Broken Tower" from others).
- Use numeric or quantitative data critically; evaluate whether numbers in the claim are supported, exaggerated, or contradicted by evidence.
- For claims referencing relationships (e.g., a person’s career influencing their work), highlight whether the passage explicitly confirms or negates this.
- If no direct evidence connects a claim element (such as a title or name) to the stated fact, clearly state the absence of support, noting relevant related details in the passages.
- When available data matches or partially matches the claim, accurately describe the level of support (e.g., “partially true,” “false in magnitude,” etc.).
- Summaries should be crafted so they can effectively guide subsequent information retrieval or query generation for more evidence, hence linking claim terms and evidence clearly and explicitly is essential.
- Avoid introducing external knowledge beyond what is logically inferable from the passages unless specifically stated.
- When quantifying differences or comparisons (e.g., population sizes), critically analyze the plausibility and mention numeric relationships rather than absolute values when evidence is lacking.

Overall, your output should demonstrate careful reasoning that draws out the relevant factual connections between the claim and the evidence passages, producing an informative yet focused summary useful for further verification or querying.
```

### Stage 3: `summarize2.predict`
- Signature: `StringSignature(claim, context, passages -> reasoning, summary
    instructions='You are given three inputs: `claim`, `context`, and `passages`.\n\n- `claim` is a stated factual assertion related to people, films, dates, or biographical details.\n- `context` briefly states whether the claim can be confirmed, refuted, or is unsupported, based on the given passages.\n- `passages` is a list of textual excerpts containing domain-specific factual information such as film titles, release years, director and writer names, actor biographies, birthdates, film relationships (e.g., sequels), and related details.\n\nYour task is to generate a concise `summary` field that synthesizes relevant evidence or lack thereof from the passages to clearly indicate whether the claim is supported or unsupported, addressing any ambiguities or missing information. The summary must:\n\n- Incorporate explicit evidence from the passages that directly or indirectly addresses the claim, including:\n  - film titles and release years,\n  - names of directors, writers, or actors,\n  - birthdates or biographical details when relevant,\n  - relationships between films (e.g., sequels or original works),\n  - geographic or origin details linked to persons if mentioned.\n- Highlight when evidence is insufficient or missing for certain aspects of the claim.\n- Explicitly connect key entity names or concepts mentioned in the claim with corresponding evidence from the passages, even if indirectly (for example, linking alternate name forms like "Rupert" vs. "Robert").\n- Use precise fact-based language rather than vague or generalized statements.\n- Avoid introducing information not present in the passages.\n- Ensure the summary could be used as a basis for generating targeted queries to relevant documents to verify or refute the claim further.\n\nIn your reasoning before writing the summary (for yourself, though not required to be output), carefully and systematically:\n\n1. Identify all entities and film titles mentioned in the claim.\n2. Search the passages for explicit or implicit evidence of these entities or titles, including possible variations or related facts.\n3. Determine if the passages provide clear support, contradiction, or no information regarding each main point.\n4. Consider domain-specific naming conventions or relationships relevant to film and biographical data (e.g., directors of films, birthdates relative to other persons, sequels).\n5. Formulate your summary to explicitly reference evidence and gaps, clearly tying back to the original claim.\n\nThis approach ensures your summary is factual, evidence-based, and useful for generating further information retrieval steps to validate the claim.'
    claim = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'})
    context = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Context:', 'desc': '${context}'})
    passages = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    summary = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'})
)`
- Fields: `{"claim": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Claim:', 'desc': '${claim}'}", "context": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Context:', 'desc': '${context}'}", "passages": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Passages:', 'desc': '${passages}'}", "reasoning": "annotation=str required=True json_schema_extra={'prefix': \"Reasoning: Let's think step by step in order to\", 'desc': '${reasoning}', '__dspy_field_type': 'output'}", "summary": "annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Summary:', 'desc': '${summary}'}"}`
- Instruction length: 2609 chars

Instruction (verbatim):

```text
You are given three inputs: `claim`, `context`, and `passages`.

- `claim` is a stated factual assertion related to people, films, dates, or biographical details.
- `context` briefly states whether the claim can be confirmed, refuted, or is unsupported, based on the given passages.
- `passages` is a list of textual excerpts containing domain-specific factual information such as film titles, release years, director and writer names, actor biographies, birthdates, film relationships (e.g., sequels), and related details.

Your task is to generate a concise `summary` field that synthesizes relevant evidence or lack thereof from the passages to clearly indicate whether the claim is supported or unsupported, addressing any ambiguities or missing information. The summary must:

- Incorporate explicit evidence from the passages that directly or indirectly addresses the claim, including:
  - film titles and release years,
  - names of directors, writers, or actors,
  - birthdates or biographical details when relevant,
  - relationships between films (e.g., sequels or original works),
  - geographic or origin details linked to persons if mentioned.
- Highlight when evidence is insufficient or missing for certain aspects of the claim.
- Explicitly connect key entity names or concepts mentioned in the claim with corresponding evidence from the passages, even if indirectly (for example, linking alternate name forms like "Rupert" vs. "Robert").
- Use precise fact-based language rather than vague or generalized statements.
- Avoid introducing information not present in the passages.
- Ensure the summary could be used as a basis for generating targeted queries to relevant documents to verify or refute the claim further.

In your reasoning before writing the summary (for yourself, though not required to be output), carefully and systematically:

1. Identify all entities and film titles mentioned in the claim.
2. Search the passages for explicit or implicit evidence of these entities or titles, including possible variations or related facts.
3. Determine if the passages provide clear support, contradiction, or no information regarding each main point.
4. Consider domain-specific naming conventions or relationships relevant to film and biographical data (e.g., directors of films, birthdates relative to other persons, sequels).
5. Formulate your summary to explicitly reference evidence and gaps, clearly tying back to the original claim.

This approach ensures your summary is factual, evidence-based, and useful for generating further information retrieval steps to validate the claim.
```

---
