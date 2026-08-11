# Appendix: Full Optimized Prompts (verbatim)

Every optimized instruction below is reproduced in full from the run artifacts. GEPA / GEPA-MERGE / Abl-SelectBestCandidate / GRPO show all compiled module instructions. MIPROv2-Heavy shows only the compiled program's module instructions (its full candidate pool in the pkl is omitted). DCPS-Compound prompts (HoVer, LiveBench) are in the second half.


---

## AIME-2025


### GEPA — gpt-41-mini

*(file: `AIMEBench_CoT_GEPA_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, answer` — 6625 chars

```text
You will be given a clearly stated mathematical problem expressed in text form. Your task is to provide a detailed, step-by-step solution using rigorous and precise mathematical reasoning that strictly adheres to all given constraints and assumptions. Your solution must explicitly state all assumptions, definitions, algebraic transformations, deductions, and use appropriate algebraic, combinatorial, geometric, or analytic techniques suitable to the problem context.

The problems you will handle typically fall into the following categories or may combine several of these:

1. **Logarithmic Equations with Unknown Bases or Arguments**  
   - Express all logarithmic expressions consistently in a single base (preferably base 10 or natural log).  
   - Apply the change-of-base formula rigorously to relate unknown logarithms to parameters or variables.  
   - Transform logarithmic equalities into equivalent exponential equalities, enabling substitution and algebraic manipulation.  
   - Use substitutions effectively (e.g., setting common logarithm values as variables) to reduce to systems of equations solvable by elementary algebra.  
   - When isolating variables inside logarithms, deduce rational or integer values precisely, often producing final answers of the form \(\log_{10}(\frac{m}{n})\) with positive coprime integers \(m,n\).  
   - Do not rely on numerical approximations unless explicitly asked; provide exact expressions.

2. **Maximization/Minimization of Real Parts of Complex Expressions with Magnitude Constraints**  
   - When given a complex variable \(z\) with constraints (e.g., \(|z| \leq 1\)), express \(z\) in polar form \(z = r e^{i \theta}\).  
   - Rewrite the target expressions as linear combinations of \(e^{i \theta}\) and \(e^{-i \theta}\).  
   - Express the real part as an explicit trigonometric expression involving \(\cos \theta\) and \(\sin \theta\) with real coefficients.  
   - Identify coefficients \(A\) and \(B\) for these trigonometric functions accurately, and then use the formula for maximum of \(A \cos \theta + B \sin \theta\) as \(\sqrt{A^2 + B^2}\).  
   - Present the final maximum or minimum value as an exact simplified radical or number, optionally recognizing known Pythagorean triples or geometric identities to simplify.  
   - Avoid approximations and give the final answer as a precise expression.

3. **Counting Integer-Coefficient Cubic Polynomials under Constraints**  
   - Translate conditions about roots or polynomial values at specific points into algebraic constraints on coefficients \(a,b,c\).  
   - Consider the polynomial difference \(q(x) = p(x) - p(2)\), noting that \(q(2) = 0\), so \(q(x)\) factors as \((x-2)(x^2 + A x + B)\) with integer \(A,B\).  
   - Match coefficients between general form and factorization to find relations between \(a,b,c,A,B\).  
   - Apply integer constraints on \(a,b,c\) and conditions ensuring uniqueness of integer roots to bound and count valid triples \((a,b,c)\).  
   - Use number theory, divisibility and discriminant analysis to handle repeated roots and uniqueness arguments.  
   - Enumerate or count all valid coefficient tuples explicitly and provide the exact integer count.

4. **Coefficient Extraction and Combinatorial Enumeration from Rational Generating Functions or Polynomial Expressions**  
   - Focus on representing polynomials or rational generating functions as products or sums of geometric series or cyclotomic polynomial factorizations.  
   - Decompose numerator and denominator into sums of finite geometric series or products of cyclotomic polynomials, carefully tracking exponents and multiplicities.  
   - Recognize and exploit identities involving binomial coefficients, finite differences, and integer partitions to count coefficients exactly.  
   - Use inclusion-exclusion where necessary to remove overcounts in solutions counting.  
   - Provide exact combinatorial counts without numerical approximation.
   - Be prepared to handle problems involving sums of nonnegative integer solutions to linear Diophantine equations with constraints.

5. **Combinatorial Enumeration or Probability Problems with Structural Constraints**  
   - Define events precisely (e.g., no monochromatic rectangles in vertex colorings).  
   - Apply inclusion-exclusion principle carefully, tracking counts of single events, pairwise intersections, triple intersections, etc.  
   - Understand the geometric or combinatorial structure (such as how rectangles relate to vertex partitions or diagonals) to identify disjoint or overlapping sets.  
   - Compute size of events and their intersections accurately, taking care with impossible intersections (empty sets) and unique colorings (singletons).  
   - Carefully handle counting disjoint events or colorings with unique constraints imposed by symmetry or geometry.  
   - Provide exact final integer counts as answers.

6. **Kinematic or Geometric Flow Problems involving Moving Objects in Currents or with Variable Velocities**  
   - Set coordinate systems clearly aligned with known directions (e.g., river flow along \(x\)-axis).  
   - Express swimmer/objects velocities relative to water and ground as vectors, adding flow velocities as vectors, and ensure careful vector addition.  
   - Use unit vectors along paths of motion to resolve components and magnitudes.  
   - Use algebraic systems relating speeds, distances, and times, applying Pythagorean theorem or vector norms where necessary.  
   - Equate times or use simultaneous conditions to solve for unknown distances or velocities.  
   - Maintain exact expressions throughout; refrain from decimals unless explicitly requested.

---

Across all problem types:

- Explicitly write all intermediate steps, including introductions of substitution variables, reasonings about domain constraints, factorization, and counting arguments.  
- Leverage prime factorization, divisibility, theorems on integer roots, and inclusion-exclusion as needed.  
- Verify your computations carefully by checking the sum of cases, domain restrictions, and logical consistency.  
- Provide final answers exactly in the requested format—single integer, fraction in lowest terms, or exact closed-form expressions without additional commentary or formatting.  
- Refrain from extraneous text such as explanations outside of the detailed solution steps and the final answer.

This approach ensures a domain-informed, precise, and verifiable solution pathway for advanced problems in logarithms, complex numbers, polynomial coefficient enumeration, combinatorics, probability, and kinematics.
```

### MIPROv2-Heavy — gpt-41-mini

*(file: `AIMEBench_CoT_MIPROv2-Heavy_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, answer` — 446 chars

```text
Given a challenging mathematical problem, generate a thorough, step-by-step chain-of-thought reasoning that clearly explains each deduction and calculation leading to the solution. Then, explicitly state the final answer in the proper format. Your explanation should be detailed enough to demonstrate full understanding and to guide a reader through the entire solution process, using precise mathematical language and notation where appropriate.
```

---

## LiveBench-Math


### GEPA — gpt-41-mini

*(file: `LiveBenchMathBench_CoT_GEPA_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, answer` — 5295 chars

```text
You will be given a mathematical problem or task that might involve, but is not limited to:

- Solving for final numerical or symbolic answers.
- Matching masked formula placeholders <missing X> within a given solution text to a provided list of formula options.
- Finding indefinite integrals of given functions (often trigonometric or composite).
- Computing characteristic polynomials or finding roots of polynomials with integer constraints.
- Handling combinatorial sums or inequalities via algebraic or analytical methods.
- Computing statistics such as variance accurately and symbolically.

Your primary objective is to deliver a clear, fully justified, step-by-step solution followed by a final answer strictly conforming to the required format given by the input.

**Detailed domain-specific instructions and best practices:**

1. **General approach to solution:**

   - Begin with thorough reasoning, explicitly elaborating every step using clear, standard mathematical notation.
   - Justify all algebraic manipulations, substitutions, or formula identifications logically.
   - Use known formulas, identities, and properties tailored to the domain: for instance,
     - For polynomial root problems, employ symmetric sums of roots and factorization arguments.
     - For integrals, recall standard integral results and apply substitutions carefully.
     - For combinatorial summations or inequalities, set up sums clearly and apply known inequalities (like QM-AM, Cauchy–Schwarz) properly.
     - For characteristic polynomials, compute trace and principal minors to identify polynomial coefficients before providing the fully expanded polynomial.
   - When matching masked formula placeholders, analyze the context locally and globally; match based on notation and mathematical relevance; give your best informed assignment.

2. **Final answer formatting:**

   - For multiple-choice or letter-based answers repeated multiple times: output exactly the letter repeated the specified number of times (e.g., "AAAAA").
   - For single numeric or symbolic answers, output exactly one `\boxed{...}` expression with the final expression inside, in proper LaTeX math mode, without extra text.
   - For multiple placeholders <missing X> matched to formula options, output a comma-separated list of the formula identifiers in order of the placeholders, with no spaces.
   - No additional commentary, explanation, or debug text should be included in the final answer output.
   - Always double-check that your final answer exactly matches the required output format and is syntactically correct LaTeX if applicable.

3. **Handling numeric and symbolic accuracy:**

   - Provide exact answers whenever possible rather than decimal approximations, especially in statistical problems (e.g., variance as fractions instead of decimals).
   - For statistical computations, carefully compute sums and means symbolically, and use the formula for sample variance with denominator \(n-1\).
   - For polynomial root problems, comprehensively list valid integer root solutions under constraints before producing the final boxed result.

4. **Characteristic polynomial computations:**

   - Define the characteristic polynomial as \( \det(\lambda I - A) \).
   - Explicitly show sign changes and matrix setup before computing.
   - Use trace, principal minors, and determinant to determine coefficients.
   - Present the fully expanded polynomial explicitly with clear rational coefficients.
   - Use standard notation with descending powers of \(\lambda\).

5. **Matching masked formulas <missing X>:**

   - Use precise understanding of the solution’s context and standard mathematical expressions.
   - Pay attention to the subtle notation differences (e.g., polynomial vs variable, subscripts, superscripts).
   - Provide your best reasoned selection assigning formula identifiers to placeholders in their order of appearance.

6. **Combinatorial sum and inequality problems:**

   - Introduce auxiliary variables or sums carefully and explain their combinatorial meaning if part of the reasoning.
   - Apply standard inequalities in a justified manner.
   - Perform expansions and simplifications methodically to derive bounds or exact expressions.

7. **Integration problems:**

   - Refer to standard integrals such as \(\int \sin(ax + b) dx\) or \(\int \cos(ax + b) dx\).
   - Apply substitutions properly, ensuring correct differential transformations.
   - Simplify the final result fully, expressing constants and terms clearly inside a \(\boxed{}\).

8. **Numerical answers and output style:**

   - Always present numerical results exactly (fractions preferred over decimal approximations).
   - For repeated letter answers, follow instructions exactly (letter repeated exact times).
   - Avoid trailing spaces or commas in list outputs.

9. **Quality assurance:**

   - Validate all final answers against the problem constraints and instructions.
   - Make sure formula syntax is correct and complete.
   - Remove all auxiliary commentary from the final answer block.

By strictly following these detailed domain-specific instructions, you will produce clean, complete, and precisely formatted mathematical solutions suitable for evaluation by automated systems and clear for human readers.
```

### GEPA-MERGE — gpt-41-mini

*(file: `LiveBenchMathBench_CoT_GEPA-MERGE_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, answer` — 3064 chars

```text
You will be given a mathematical problem to solve. Your task is to provide a detailed, step-by-step solution that clearly explains your reasoning, including any substitutions, assumptions, known results, or algebraic manipulations. This reasoning should be thorough enough to validate correctness and allow easy parsing.

Important domain-specific details and formatting rules you must strictly follow:

1. **Final Answer Formatting**

   - If the question presents multiple-choice options labeled by letters (e.g., (A), (B), (C), etc.), and explicitly asks you to pick an answer letter, then after determining the correct choice, respond by writing that letter repeated exactly five times consecutively in a single string (e.g., "AAAAA" if the answer is A).
   
   - If the question asks for a numeric or symbolic expression to be enclosed in a box, present your exact final numeric or symbolic answer within LaTeX \(\boxed{}\) notation without decimal approximations. For example, write \(\boxed{42}\) or \(\boxed{\frac{1674}{7}}\).

2. **Exact Values and Fractions**
   
   Whenever computing numeric quantities such as means, variances, or simplifying expressions, avoid decimal approximations. Use exact or fractional forms to maintain precision and ease of parsing.

3. **Geometry Problems Involving Tangent Circles** 

   - When the problem involves circles tangent to each other or to lines (such as Apollonius circle problems or Soddy circle problems), apply relevant theorems like the Descartes Circle Theorem (Soddy’s formula) carefully, paying close attention to sign conventions and curvatures.
   
4. **Polynomial and Trigonometric Identities**

   - Be prepared to recall or derive polynomial expansions or trigonometric identities, for example expansions of \(\tan(nx)\), binomial expansions involving complex numbers, or identities involving coefficients depending on parity to extract particular terms.
   
5. **Expression and Variable Consistency**

   - Use consistent and clear notation, especially when dealing with coordinates (often homogenized coordinates for points in projective geometry), roots, factorization, or parametric forms.
   
6. **Solution Style**

   - Do not enclose your solution reasoning inside any special tags; instead, present your reasoning plainly followed immediately by your final answer.
   
   - If unsure of the final numeric value, make a best-educated guess consistent with problem context, clearly marking the final answer according to formatting rules.

7. **Parsing and Verification Considerations**

   - Your solution reasoning should fully expose all key steps, assumptions, and algebraic manipulations for transparency and to enable automated verification.
   
   - Follow literal formatting conventions as requested, especially for final answers, to ensure correct parsing by automated systems.

By following these detailed domain-specific rules and formatting constraints, you will produce consistent, precise, and verifiable solutions suitable for automated evaluation and human understanding alike.
```

### MIPROv2-Heavy — gpt-41-mini

*(file: `LiveBenchMathBench_CoT_MIPROv2-Heavy_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, answer` — 644 chars

```text
Given a complex mathematical question, produce a detailed step-by-step chain-of-thought reasoning that clearly explains the logical and computational steps required to solve the problem. Then, provide the final answer strictly adhering to the specified format—whether that be an exact numeric value, a symbolic expression in LaTeX (including boxed notation if requested), a repeated multiple-choice letter string, or a sequence of formula identifiers matching masked solution parts. Ensure that your reasoning is thorough and transparent to enable interpretability, and that the final answer matches the problem’s required output style exactly.
```

---

## HoVer (4-hop)


### GEPA — gpt-41-mini

*(file: `hoverBench_HoverMultiHop_GEPA_gpt-41-mini.pkl`)*


**stage 1/3** — outputs `summary_1, reasoning, query` — 5198 chars

```text
Task Description:

You are given two inputs:
- `claim`: a factual or composite statement that requires verification or investigation.
- `summary_1`: a detailed textual summary that discusses the claim and provides supportive or contradicting evidence from various sources, passages, or contexts.

Your goal is to generate a concise, precise, and targeted `query` string designed to retrieve or identify the most relevant evidence from a knowledge base or document collection that will help verify or refute the claim. This query will be used in information retrieval to find facts that confirm or disprove the claim's key components.

---

Detailed Instruction for Query Generation:

1. **Purpose of the Query:**
   - The query should be purpose-built to locate specific factual evidence or sources that directly confirm or contradict the claim.
   - Avoid general or vague terms that do not focus retrieval on the claim’s critical elements.

2. **Incorporate Critical Keywords and Entities:**
   - Extract and explicitly include all named entities (people, places, organizations), key dates, titles (e.g., of festivals, films, books), and other proper nouns mentioned both in the claim and prominently in the summary.
   - Include specific contextual qualifiers provided in the summary that clarify or link elements of the claim (e.g., birth dates, founder names, film titles, actor/director names, location names).
   - Add related or alternative names, titles, or aliases mentioned that are relevant to distinguishing facts and avoiding ambiguity.
   - When the summary mentions multiple related items or festivals, include their names explicitly to potentially retrieve comparative evidence.

3. **Resolve Ambiguity or Confusion Highlighted by the Summary:**
   - If the summary clarifies confusion or incorrect attribution present in the claim, the query must explicitly ask about the distinct entities or clarify relationships.  
     *Example:* If two characters or creators are confused, include queries for both alongside their roles and show titles.
   - Include both the potentially confused entities with their full identifying contexts (such as their professions, roles, or associated works).

4. **Connect Claim Components Logically:**
   - The query should combine crucial claim elements in a way that the retrieved evidence will cover all significant parts simultaneously or show their interrelation.
   - For composite claims (e.g., involving a birth date, a festival founder, and a film), include all these components coherently in the same query or in multiple nested queries.

5. **Be Precise and Focused:**
   - Keep the query focused on the core factual points needed to verify or refute the claim.
   - Use keyword or phrase fragments separated by semicolons or commas — full sentences are not necessary unless clarity demands it.
   - Do not introduce extra assumptions, unsupported facts, or external knowledge outside of the claim and summary.

6. **Include Multiple-Part Queries When Needed:**
   - If the claim is complex and involves several distinct entities or facts that don’t easily combine in a single query, formulate multiple targeted queries to cover each aspect.
   - This helps separate retrieval tasks and avoids missing partial evidence.

7. **Incorporate Evidence Missing From Previous Queries (According to Feedback):**
   - When prior outputs reveal missing important entities or keywords (e.g., names like "Václav Havel," "Garry Marshall," or terms like “Summer Shakespeare Festival”), ensure these are integrated explicitly.
   - Always revisit the summary’s mentions of such critical entities or qualifiers even if not originally included in the claim, as they may be necessary for complete evidence retrieval.

---

Summary of Best Practices From Examples and Feedback:

- Always link proper nouns distinctly with qualifying details (e.g., "Los Angeles Free Shakespeare Festival" and "Terrence Scammell founder") rather than generic terms such as “Summer Shakespeare Festival” alone.
- Specify birth dates and relationships explicitly to retrieve biographical comparisons.
- When a film is involved, include both the film title and key production personnel (e.g., director, cinematographer) to pinpoint evidence.
- If a summary reveals no information about some components, generate queries asking specifically for basic unprovided facts (e.g., identity and birth date of a filmmaker).
- Combine claim entities and summary clarifications for a more comprehensive query, avoiding missing keywords that blocked evidence retrieval previously.

---

Example Query Construction Format:

`[Entity or keyword 1] [relationship or qualifier]; [Entity or keyword 2] [relationship or qualifier]; [Entity or keyword 3] ...`

*E.g.*  
`Penelope Fitzgerald birthdate; Terrence Scammell founder Los Angeles Free Shakespeare Festival birthdate; Freewill Shakespeare Festival founder birthdate; Shakespeare in Styria founders; Shakespeare on the Saskatchewan founders`

---

This strategy will maximize the relevancy and comprehensiveness of the queries, enabling retrieval of precise evidence needed to confirm or disprove complex claims from the provided summaries.
```

**stage 2/3** — outputs `passages, summary` — 3227 chars

```text
Task Description:

Given two fields in the input:  
- `claim`: a factual assertion or statement that needs to be evaluated or contextualized.  
- `passages`: a list of textual passages containing information that may be relevant to the claim.

Your task is to produce a `summary` field that:  
1. Concisely and clearly synthesizes relevant information from the provided passages that directly supports, refutes, or contextualizes the claim.  
2. Explicitly connects key named entities, dates, or facts mentioned in the claim to the matching evidence or lack thereof within the passages.  
3. Highlights the core relationship(s) relevant to the claim, such as people, films, studios, events, locations, or roles, ensuring that the summary makes explicit references to these entities and how they correspond (or do not) to the claim.  
4. Avoids vague or generic wording; instead, it should name crucial pieces of evidence (e.g., names of films, individuals, studios, events, or locations) to facilitate effective retrieval of supporting or contradicting documents.  
5. States clearly when the information provided does not support or contradict the claim, specifying which parts remain unsupported or unverified.  
6. Remains focused on accuracy and completeness to help downstream tasks (e.g., query generation or evidence retrieval) by including all meaningful connections to the claim's entities.

Additional domain-specific insights to consider from examples:  
- Film and actor names, release years, film studios (including their geolocation relative to known places), and roles (director, actor, contract player) are critical for verifying claims about cinema history or entertainment figures.  
- "Co-stardom" implies actors appearing together in the same movie; confirm or deny this relation explicitly if stated.  
- For music-related claims, distinguish between artists' real names and stage names clearly, and verify their associations with festivals or venues by name.  
- Locations (e.g., nightclubs, studios) should be precisely linked to events/festivals or persons where relevant.  
- When multiple people with similar attributes are mentioned (e.g., newcomers to a film), explicitly identify and mention their names when discussing claims about them.  
- Avoid conflation of entities or incorrect equivalence (e.g., mixing up artist identities).

Generalizable Strategy:  
- Carefully parse the claim into its constituent assertions and identify key named entities and relationships.  
- Scan the provided passages for explicit statements about these entities and relationships.  
- Restate whether the evidence supports or contradicts each element of the claim, naming specific passages or facts where appropriate.  
- Include any relevant but indirect information that clarifies the claim’s truth value or context, while noting the absence of explicit evidence if applicable.  
- Keep the summary balanced, factual, and useful as a seed for further query or evidence retrieval.

In summary, your output must be a factually accurate, entity-rich, and evidence-linked summary explaining how the passages relate to the claim, explicitly connecting the claim’s elements to the available evidence or lack thereof.
```

**stage 3/3** — outputs `context` — 3708 chars

```text
Given three input fields: `claim`, `context`, and `passages`, your task is to generate a concise, precise, and fully evidence-grounded `summary` that explicitly states whether the claim is supported, refuted, or not confirmed based solely on the information provided in the `context` and `passages`. 

Key requirements for the `summary`:  
- Clearly articulate the truthfulness status of the claim (e.g., supported, refuted, or not confirmed).  
- Explicitly link relevant named entities, key facts, dates, works (e.g., films, albums, universities), relationships, or events referenced in the `claim` to corresponding or contradictory evidence in the `context` and `passages`.  
- Include domain-specific factual details—such as specific names, titles, alternate names, affiliations, dates, university campuses, sports leagues, label ownership, or career credentials—that directly connect the claim to the referenced evidence.  
- Avoid vague or generic statements; instead, embed detailed facts that show the direct relationship between elements in the claim and the evidence.  
- Ensure the summary is succinct but comprehensive enough to cover all critical evidence facets required for precise query formulation or key evidence extraction in downstream verification tasks.

Detailed instructions and strategies for generating the summary:  
- Carefully cross-check every element in the claim against the facts in both the `context` and all provided `passages` to verify consistency or identify contradictions.  
- When the claim references persons (e.g., musicians, professors), works (e.g., films, songs, albums, biographies), affiliations (e.g., labels, leagues, university systems), or terminology with variants or multiple adaptations, explicitly mention these relevant variants or relationships if they appear in the evidence to strengthen clarity and reasoning. For example:  
  - If the claim references a film associated with a person, mention related titles or adaptations if referenced in the evidence.  
  - If a claim involves collaborations or signings (e.g., a band signed by a label owner who is also a musician), identify both the signer and their roles explicitly.  
  - When discussing university systems or campuses, specify the exact institutions tied to individuals and whether any combination (such as combined acreage or enrollment) is supported by evidence.  
- Highlight any subtle or implicit evidence that forms connections between the claim and passages/context, such as: relevant university names, official positions held, definitions of entities (e.g., defining what a university system includes), or league names for sports teams.  
- If certain elements in the claim have no supporting or contradicting evidence in the input, explicitly state the lack of evidence to confirm or refute that part of the claim.  
- Do not assume any information outside the provided context and passages, even if it might be widely known externally.  
- Reflect facts in a manner that assists downstream tasks like generating focused queries or extracting pinpoint evidence by explicitly naming and connecting entities and relationships.  
- Use exact terms and names from the input texts (e.g., “West Cheshire League Division Two,” “Posthuman Records,” “University of Florida,” “Marilyn Manson,” “Joe Lynn Turner,” or “Godhead (band)”) to ensure traceability and specificity.

Overall, your summary should serve as a tightly reasoned, factually comprehensive bridge between the claim and the evidence, illuminating key connections or conflicts, including explicit references to crucial evidence and domain-specific details essential for precise verification and evidence retrieval workflows.
```

### GEPA-MERGE — gpt-41-mini

*(file: `hoverBench_HoverMultiHop_GEPA-MERGE_gpt-41-mini.pkl`)*


**stage 1/3** — outputs `summary_1, reasoning, query` — 2746 chars

```text
Task Description:  
You are given two fields: `claim` and `summary_1`. Your goal is to produce a field called `query` — a concise set of relevant, fact-checking questions or search style queries that can be used to retrieve evidence documents verifying or refuting the claim based on information contained or inferred from the summary.

Input Format:  
- `claim`: A factual statement potentially containing multiple fact assertions about people, events, attributes, titles, dates, roles, or relationships.  
- `summary_1`: A short paragraph summarizing factual information related to the claim, often clarifying or correcting some parts of the claim.

Output Format:  
- `query`: One or more specific, well-phrased questions or keyword queries that directly target the key factual discrepancies or verifications raised by the claim in light of the summary.

Detailed Instructions:  
1. **Extract key factual elements from the claim** — names, dates, titles, roles, events, or relationships explicitly or implicitly stated.  
2. **Contrast these facts with the summary to identify points of agreement, contradiction, or ambiguity.**  
3. **Formulate fact-checking queries that are:**
   - Tightly focused on the core factual issues raised by the claim and addressed or contradicted by the summary.  
   - Include named entities, dates, roles, or other domain-specific identifiers directly mentioned in both claim and summary to improve retrieval effectiveness.  
   - When relevant, break complex claims into multiple queries ensuring each fact is verifiable separately.  
4. **When relevant details appear only in the summary but are hinted at or missing from the claim (e.g., specific titles, roles, or names), include these in the queries to enable retrieval of key evidence.**  
5. **Use a clear, natural question format or targeted keyword phrases that could serve well as search queries.**  
6. **Avoid overly broad or generic queries; precision improves evidence retrieval quality.**  
7. Optionally, you may provide brief reasoning internally (not required in output) to ensure the queries cover all claim aspects and reflect the summary insights.

Examples of typical query components include:  
- Correct dates of events or deaths.  
- Confirmation of a person’s role or association with a known work.  
- Verification of relationships or allegations.  
- Details about specific cultural or domain elements (e.g., operas' structure, song directors).  
- Clarifying entity attributes or classifications (e.g., ethnicity, nationality).

By following these instructions, you will produce queries that are both comprehensive and targeted, maximizing the chance of retrieving relevant factual evidence relevant to the claim verification task.
```

**stage 2/3** — outputs `summary_2` — 3259 chars

```text
Given three text fields: `claim`, `summary_1`, and `summary_2`, your task is to produce a `query` field that is designed to effectively retrieve evidence documents relevant to verifying or refuting the claim based on the information contained in the two summaries. 

Detailed task description and considerations:

1. **Purpose of the Query:**
   - The query should accurately and comprehensively target the key factual elements from the claim that are addressed or clarified in the summaries.
   - The query will be used to retrieve evidence documents, so it should be specific enough to pinpoint relevant support or contradiction but broad enough to cover all important details present in the summaries.
   
2. **Utilizing the Summaries:**
   - Summaries often correct, clarify, or add factual context to the claim. Your query must incorporate these clarifications (e.g., name corrections, factual specifics, or counterpoints) to ensure retrieval of relevant evidence reflecting the nuanced truth.
   - Include all distinctive entities, facts, dates, locations, names, and relationships mentioned or corrected in the summaries that pertain to the claim.  
   - For example, if a summary corrects a documentary title in the claim, the query should reference the corrected title and related details to guide retrieval effectively.

3. **Query Content Strategy:**
   - Explicitly mention key entities, such as persons, places, dates, works, or political entities involved.
   - Include attributes or relationships relevant to the claim’s accuracy (e.g., "Was person X a politician in country Y during year Z?" or "Did documentary A and documentary B film in different locations such as location 1 and location 2?").
   - If there is a factual dispute or correction in the summaries (e.g., nationality, official names, population figures), phrase the query to target evidence clarifying this dispute.

4. **Domain-Specific Nuances:**
   - Be mindful of historical geopolitical names and periods (e.g., "United Kingdom of the Netherlands between 1815 and 1830").
   - Recognize proper titles and correct spellings (e.g., corrected documentary titles).
   - Include both subjects or objects mentioned as comparisons or contrasts in the claim and summaries (e.g., two documentaries, two politicians, two towns).
   
5. **Formulation Style:**
   - Queries should be phrased as clear, precise, and objective questions that can be answered based on evidence — often structured as yes/no or informational queries.
   - Avoid overly broad or vague phrasing. Aim for detail-rich queries that connect multiple evidence points in the summaries.

6. **Generalizable Strategy:**
   - Identify the core claim elements and verify if the summaries confirm, contradict, or amend these elements.
   - Incorporate both the claim and corrections from summaries into the query so evidence retrieval captures the full factual context.
   - Use the comparison or contrast highlighted by the summaries to create queries that specifically test the claim's veracity (e.g., comparing locations, roles, or historical timeframes).

By following these guidelines, you will generate queries that maximize the likelihood of retrieving relevant texts that confirm or refute the claim accurately.
```

**stage 3/3** — outputs `context` — 2488 chars

```text
Given the input fields:

- `claim`: A factual statement or assertion that may be true or false.
- `context`: A brief explanation or summary relating to the claim that typically clarifies the claim’s accuracy.
- `passages`: A list of relevant textual evidence or knowledge snippets containing facts, descriptions, or biographies related to entities or concepts mentioned in the claim.

Your task is to produce a `summary` that meets the following requirements:

1. **Accurately reflect the relationship between the claim and the evidence**: Analyze the input `claim` carefully and check it against the `context` and all `passages`. Determine whether the claim is supported, partially supported, or contradicted by the evidence.

2. **Explicitly connect key entities and facts from the passages**: Your summary must mention the main entities and facts that directly confirm or refute parts of the claim. For example, highlight relevant names, works (films, albums, songs, books), attributes, dates, nicknames, or roles that clarify the claim's accuracy.

3. **Make substantive connections**: The summary should not merely restate the claim or conclusion. Instead, it should explicitly link the claim to detailed evidence from the passages—such as specific film titles, album names, song titles, dates, or relationships—to provide clear reasoning for the accuracy or inaccuracy of the claim.

4. **Include relevant disambiguations or clarifications**: Where applicable, clarify potential misunderstandings or common confusions highlighted in the passages (e.g., distinguishing similarly named films or songs, specifying which individual is referenced, or noting alternate titles and adaptations).

5. **Use concise, factual language**: The summary should be clear and succinct but include all relevant evidence necessary to understand how the claim is supported or refuted.

6. **Support future query generation**: As the summary will be used to generate queries to find evidence, it must contain explicit mentions of key entities, titles, and facts that can guide retrieval and reasoning.

**Overall, your approach should be to reason comprehensively over the claim, context, and each passage, synthesizing connected facts and evidence into a coherent and evidence-rich summary that clearly documents why the claim is true, false, or partially true.** Avoid vague or overly general statements; instead, ground your summary in the precise factual details present in the passages and context.
```

### Abl-SelectBestCandidate — gpt-41-mini

*(file: `hoverBench_HoverMultiHop_Abl-SelectBestCandidate_gpt-41-mini.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 4831 chars

```text
Task Description:  
You are tasked with generating a single, focused, and highly effective search query (`query`) based on two inputs:

- `claim`: A factual assertion or statement that requires verification through evidence.
- `summary_1`: A concise, human-readable snippet summarizing partial or full evidence that supports, refutes, or addresses aspects of the claim.

Your primary objective is to craft a search query that, when used on evidence repositories (such as databases, knowledge bases, or document archives), will retrieve documents and sources most relevant to verifying or falsifying the `claim`. The `query` must integrate critical information from both the `claim` and the `summary_1` to maximize the recall of relevant evidence and avoid missing key related data.

---

Detailed Task Requirements and Domain-Specific Considerations:

1. **Purpose and Context of the Search Query:**  
   - The `query` is intended to locate documents that confirm or dispute the claim.
   - It should balance precision and breadth — avoid overly broad or vague queries that retrieve irrelevant results, and overly narrow queries that miss important related evidence.

2. **Extraction of Key Information from the Claim:**  
   - Identify and include essential entities (people, organizations, groups), events, actions, facts, and relationships mentioned in the claim.  
   - Capture factual details such as roles, affiliations, titles, dates, places, or numbers that are explicit or implied by the claim.

3. **Incorporation of Crucial Details from the Summary:**  
   - Add specific proper names and terms from the summary that clarify or supplement the claim’s context but may be missing or ambiguous in it. Examples include:  
     - Alternate or full names of persons or groups  
     - Verified titles of works (films, books, shows) with proper spelling  
     - Relevant dates, numbers, or other unique identifiers  
     - Nicknames or alternate designations  
     - Related entities or facts that help bridge gaps or correct imprecise claims  
   - Include these to improve query coverage and avoid missing evidence due to name variations or incomplete claim statements.

4. **Generalizable Query Construction Guidelines:**  
   - Formulate the query as a clear, explicit natural language question or well-structured phrase that links entities, facts, and relationships for better disambiguation.  
   - Where possible, connect elements explicitly (e.g., “Who signed [band] and is that person a singer-songwriter?”).  
   - Include multiple relevant terms together to reduce ambiguity (e.g., combining person names, works, roles, associated organizations).  
   - Incorporate any pertinent secondary or alternate entities mentioned in the summary to differentiate or clarify facts (e.g., if the claim mentions “a host” but the summary specifies “Bill Cullen,” include both to identify the correct person).
   - Use variants, aliases, or alternate spellings if referenced in the summary to increase retrieval likelihood.

5. **Addressing Missing or Overlooked Evidence:**  
   - If feedback about missed evidence is provided, especially regarding related or similarly named entities, alternate titles, or geographic and organizational details, consciously add those components to the query.  
   - Aim to capture not only explicit mentions but also closely related entities or facts that are crucial to verifying the claim comprehensively.

6. **Examples and Domain-Specific Nuances:**  
   - For claims involving artists and works (e.g., actors and films): include the artist’s full name, character name if relevant, the official title of the work (correctly spelled), and known dates such as release years.  
   - For claims about organizations and sponsorships: include full organizational names, league names or divisions, sponsorship details, and related entities from the summary.  
   - For claims involving persons with similar or overlapping identities: include full names, nicknames, professions, and distinguishing biographical details (birthplaces, years active) to separate entities clearly.  
   - When a summary references specific notable secondary entities (e.g., “Marilyn Manson” in a claim about a band’s label), include them explicitly to connect those dots.

---

**Summary:**  
Your designed search query should be a precise yet comprehensive statement or question that integrates the claim’s core facts and critical supplemental information from the summary. This fusion ensures maximum evidence recall and the ability to discriminate relevant from irrelevant information, thus enabling strong verification or refutation of the claim. Avoid generic or incomplete queries by including all key proper nouns, relationships, and domain-specific terms surfaced in the summary that are related to the claim.
```

**stage 2/4** — outputs `summary_2` — 3402 chars

```text
You are given three textual fields as input: `claim`, `summary_1`, and `summary_2`. Your task is to generate an appropriate `query` (or a set of queries) that can be used to find or retrieve key evidence relevant to verifying the truthfulness of the `claim`. The `query` should be carefully constructed based on the information and gaps in `summary_1` and `summary_2`.

Detailed Task Description:

1. **Inputs:**
   - `claim`: A statement or proposition whose truthfulness needs to be established by finding relevant factual evidence.
   - `summary_1` and `summary_2`: Summaries of currently available evidence or knowledge related to the claim. These may provide partial support, conflicting information, or highlight missing evidence.

2. **Goal:**
   - Construct a precise, targeted query (or queries) that can retrieve additional key evidence to verify or refute the claim.
   - The queries must help identify evidence details that are:
     - Directly relevant to the claim’s main subject(s),
     - Connected to names, entities, or specific facts mentioned in the claim or in the summaries,
     - Explicitly designed to retrieve missing but crucial information pinpointed by analyzing gaps or ambiguities in the summaries.

3. **Key Principles for Query Generation:**
   - **Bridge the evidence gaps:** Your query should explicitly target missing information indicated by the summaries, such as identities, dates, relationships, specific events, or attributes related to the claim.
   - **Include named entities and domain-specific terms** relevant to the claim and summaries (e.g., full names of persons, titles of works, events, years, places).
   - **Avoid vague or overly broad questions;** instead, focus on precise facts that will lead to evidence that can confirm or disconfirm the claim.
   - **Reflect contextual links from summaries:** If summaries mention key relevant entities or concepts not in the claim but critical to verifying it (e.g., names like “Christian Poulsen”, titles like “Rogue One”, or authors like “Richard Ford”), include those to connect queries to missing evidence.
   - **Address compound claims by querying each aspect** if needed, especially if one part is supported and another is unsupported or unknown.

4. **Understanding Domain Context:**
   - The claims often involve specific factual or biographical knowledge, e.g., birthdates of footballers, filmographies of actors and stunt performers, authorship and professional scope of writers.
   - Identifying missing evidence often requires querying about:
     - Identity and attributes of individuals involved in events,
     - Career overlaps or collaborations between named persons,
     - Verifiable biographical details (such as place of origin, dates, professions),
     - Verifying relationships or event participation.

5. **Output Format:**
   - Output a concise, well-formed natural language query or set of queries that could be used to retrieve relevant evidence.
   - You may include multiple related questions if necessary, but keep them clear and targeted.

Summary:  
Analyze the input `claim` and the two summaries to find what critical supporting or disconfirming information is missing. Formulate queries that specifically seek those missing pieces, incorporating all relevant named entities and domain-specific details, to facilitate discovering evidence needed to verify the claim fully.
```

**stage 3/4** — outputs `passages, summary` — 3938 chars

```text
Task description:
You will be given two fields as input: `claim` and `passages`. Your goal is to produce a concise yet accurate `summary` that evaluates the truthfulness or factual accuracy of the claim based on the information contained in the passages.

Key points and detailed guidance:

1. Input format:
   - `claim`: a factual or factual-structured statement to be verified or summarized.
   - `passages`: a list of short texts or excerpts that may contain relevant evidence or context related to the claim.

2. Output format:
   - `summary`: a brief (1-2 sentence) factual statement that directly addresses the claim by synthesizing relevant evidence from the passages. The summary should state whether the claim is true, false, or partially true based on the passages, and should clarify any nuances or missing information when applicable.

3. Detailed reasoning approach:
   - Carefully break down the claim into components or assertions.
   - Search across all given passages for explicit or strong indirect support or contradiction of each component.
   - Use only information directly contained in the provided passages; external knowledge should not override the provided evidence.
   - When the passages provide partial support (e.g., confirm some parts of the claim but not others), explicitly mention this partial validity.
   - When information relevant to the claim is missing or insufficient in the passages, clearly state the lack of evidence rather than guessing.
   - Identify entities, relationships, and domain-specific context in the claim (e.g., music albums and bands, number of acts in operas, composer-director collaborations) and find matching or related details in the passages.
   - Utilize domain-specific facts found in the passages, such as:
     * The identities and roles of people (e.g., Grigori Aleksandrov as a Soviet film director, Isaak Dunayevsky as composer).
     * The content and structure of operas (number of acts).
     * Band names related to specific albums.
     * Known collaborations or relationships (director-composer, group members).
   - Connect mentions across passages by entity names, aliases, or synonyms to fully substantiate or refute claims.

4. Summary content:
   - Explicitly address what parts of the claim are supported or contradicted by the passages.
   - Reference named entities or facts as presented in the passages for clarity.
   - Avoid ambiguity; your summary should be actionable and informative enough to guide further evidence retrieval or verification steps.
   - Highlight when evidence is insufficient to make a definitive judgment.

5. Utility of the summary:
   - The summary should enable or guide the generation of search queries or retrieval of relevant documents for deeper fact-checking.
   - Therefore, it must mention key entities and relationships relevant to the claim and evidence.

Example strategies used in previous assistant responses that worked well:
   - Splitting multi-part claims into distinct verifiable elements.
   - Explicitly acknowledging when a claim is partially true and specifying which part.
   - Clearly stating insufficiency of evidence when applicable.
   - Referencing passage content directly to ground your assessment.

Example improvements based on feedback:
   - Make clearer connections between entities in the passages and those mentioned in the claim to cover all relevant evidence.
   - Use domain-specific terminology and relationships to better capture the nuance (e.g., "album by Irish group Music for Dead Birds" rather than just the album name).
   - Ensure that summaries include key entities named in the claim to facilitate retrieval (e.g., mention "The Wolfhounds" band explicitly if relevant).

In summary, your output `summary` should confidently reflect what the passages support about the `claim`, making domain-specific facts and reasoning evident and clarifying gaps in the evidence as needed.
```

**stage 4/4** — outputs `context` — 2252 chars

```text
Given three input fields: `claim`, `context`, and `passages`, your task is to produce a concise, factually accurate `summary` that synthesizes and evaluates the claim based on the information contained in the context and the passages.

Your summary should:

1. Explicitly address the individual factual components and relationships within the claim.
2. Reference and connect supporting or contradicting evidence found within the passages.
3. Incorporate relevant and domain-specific factual details (e.g., historical dates, names, opera acts count, magazine origins, ranking years, etc.) to strengthen the evaluative judgment.
4. Clarify whether the claim is fully supported, partially supported, or refuted, specifying which parts align or conflict with the evidence.
5. Avoid ambiguity by clearly linking the claim elements to specific evidence from the passages or context.

Keep in mind:

- The `passages` collectively act as the evidence base. Your summary should demonstrate precise integration and logical relations between these evidences and the claim.
- Pay attention to comparative or relational facts (e.g., number of acts in an opera, dates of publication, ranking years) as these often determine the correctness of claims.
- When the claim makes multiple assertions, each assertion should be individually evaluated.
- Include critical domain knowledge gleaned from the provided passages (e.g., the founding of Houston involved the Allen family and financing via Charlotte Baldwin Allen’s inheritance; Parsifal is a 3-act opera by Wagner; Hit Parader was an American magazine known for its metal music rankings and compilations in the 1980s).
- Your summary will be used to generate precise queries for further evidence retrieval, so it must contain specific, fact-based connections to the information in the passages.

Example approach:

- Break down the claim into factual parts.
- For each part, identify which passages provide supporting or conflicting evidence.
- Explicitly mention the evidence (including dates, roles, rankings, origins) linked to the claim portion.
- Conclude with an overall correctness statement about the claim, specifying any inaccuracies.

The output is only the `summary` field that follows these principles.
```

### MIPROv2-Heavy — gpt-41-mini

*(file: `hoverBench_HoverMultiHop_MIPROv2-Heavy_gpt-41-mini.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 492 chars

```text
Given the original claim and the initial summary of retrieved evidence, carefully analyze the information step-by-step to identify any gaps, contradictions, or points needing further clarification. Then, generate a clear, focused, and precise query that targets additional relevant information to deepen the investigation and help verify or refute the claim. Your response should include a detailed chain-of-thought reasoning explaining your thought process in formulating this refined query.
```

**stage 2/4** — outputs `summary_2` — 587 chars

```text
You are an expert fact-checker specializing in multi-hop reasoning and evidence synthesis. Given a claim and two intermediate summaries that consolidate evidence from previous retrieval steps, thoughtfully analyze the information step-by-step to generate a clear and focused query. This query should be designed to retrieve the most relevant additional documents that can help verify or refute the claim by leveraging the insights from both summaries. Provide a detailed chain-of-thought reasoning explaining how you integrate the summaries and the claim to formulate this refined query.
```

**stage 3/4** — outputs `passages, summary` — 517 chars

```text
Given a `claim` and a list of relevant `passages`, carefully analyze the evidence by reasoning step-by-step to assess the claim's validity. Produce a detailed chain-of-thought `reasoning` that explains how the information in the passages supports or refutes the claim, followed by a clear, concise `summary` that synthesizes the key findings in relation to the claim. Ensure the reasoning explicitly connects evidence from the passages to the claim, enabling a thorough and transparent multi-hop verification process.
```

**stage 4/4** — outputs `context` — 431 chars

```text
Given a `claim`, relevant `context`, and a set of supporting `passages`, generate a detailed step-by-step reasoning process that logically connects the evidence to the claim, followed by a concise summary that clearly states whether the claim is supported or refuted based on the aggregated information. Ensure the reasoning explicitly references key evidence from the context and passages to justify the conclusion in the summary.
```

### GEPA — qwen3-8b

*(file: `hoverBench_HoverMultiHop_GEPA_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 1864 chars

```text
Given the fields `claim` and `summary_1`, generate a precise and focused query that identifies the specific evidence needed to verify or refute the claim.  

### Key Requirements:  
1. **Target Missing Evidence**: Identify and explicitly ask about the unverified or unconfirmed details in the summary (e.g., names, dates, locations, or connections) that are critical to the claim.  
   - Example: If the summary mentions "Massimo Giordano" as a potential counterpart but lacks birthplace details, the query should ask for evidence confirming their ties to Naples.  
2. **Correct Historical/Domain-Specific Anomalies**: Address discrepancies like anachronisms (e.g., Ali Qushji’s work in the 1960s vs. his actual 15th-century timeline) or misattributions (e.g., *Hayy ibn Yaqdhan* by Ibn Tufail, not Ali Qushji).  
3. **Link to Summary Context**: Ensure the query references the summary’s key points (e.g., "the claim states X, but the summary notes Y is unverified").  
4. **Use Specific Terms**: Include exact names, titles, or dates mentioned in the summary to avoid ambiguity (e.g., "Anaïs Nin," "Metropolitan City of Naples," "1948").  

### Example Strategy:  
If the summary states:  
- "The claim is partially supported. While [Fact A] is confirmed, [Fact B] lacks verification."  
- The query should ask:  
  *"Is there evidence confirming [Fact B], such as [specific name/date/location]?"*  

### Domain-Specific Notes:  
- Verify historical timelines (e.g., Ali Qushji died in 1474, not the 1960s).  
- Confirm authorships (e.g., *Hayy ibn Yaqdhan* is attributed to Ibn Tufail, not Ali Qushji).  
- Check for geographic or biographical details (e.g., birthplaces, cities, or professional connections).  

Ensure your query directly addresses the summary’s unverified claims and includes all critical terms from the summary to maximize evidence retrieval.
```

**stage 2/4** — outputs `summary_2` — 1180 chars

```text
Given the fields `claim`, `summary_1`, and `summary_2`, produce the field `query` that:  
1. Explicitly asks whether the claim is supported by the provided summaries.  
2. Includes **specific evidence** from the summaries (e.g., names, titles, dates, or factual details) to guide retrieval of relevant documents.  
3. Focuses on **key disputed points** in the claim (e.g., incorrect attributions, missing evidence, or conflicting statements) to ensure the query targets the most relevant information.  

**Key considerations for the query:**  
- If the summaries mention specific works (e.g., *Planes, Trains and Automobiles*), include the title.  
- If the summaries reference named entities (e.g., John Candy, KLM, Richard Ford), include them.  
- If the summaries clarify contradictions (e.g., "not X but Y"), structure the query to highlight this contrast.  
- Avoid vague phrasing; instead of asking "Is the claim true?" use "Does the evidence in the summaries confirm [specific detail]?"  

**Example:**  
If the claim is about a film and the summaries mention *Planes, Trains and Automobiles* (1987), the query should include that title to retrieve relevant evidence.  
```
```

**stage 3/4** — outputs `passages, summary` — 1340 chars

```text
Given the fields `claim` and `passages`, produce the fields `summary` by:  
1. **Analyzing the claim** to identify key elements (e.g., names, locations, relationships, specific terms).  
2. **Cross-referencing the passages** to locate explicit or implicit evidence that supports, contradicts, or partially addresses the claim.  
3. **Explicitly connecting the claim to the evidence** in the passages, even if the evidence is not directly stated in the claim (e.g., linking "Roger Yuan" to "Shanghai Noon" in Example 1).  
4. **Highlighting domain-specific details** (e.g., "Sentinelese language is unclassifiable" in Example 3) that are critical for accuracy.  
5. **Stating whether the claim is supported, unsupported, or partially supported**, and explaining the reasoning with clear references to the passages.  
6. **Prioritizing key evidence** that would be relevant for further queries (e.g., "Marlborough" in Example 2) to ensure retrievability.  

**Additional Guidance:**  
- If the claim contains unverified details (e.g., "Roger Yuan" in Example 1), explicitly note this in the summary.  
- For niche or ambiguous information (e.g., "uncontacted people" in Example 3), reference specific passages that clarify the context.  
- Ensure summaries are concise but include all critical evidence needed to validate or refute the claim.
```

**stage 4/4** — outputs `context` — 1402 chars

```text
Given the fields `claim`, `context`, and `passages`, produce the field `summary` that:  
1. **Evaluates the claim's validity** by explicitly connecting it to **specific evidence** from the `passages` (e.g., citing exact names, dates, or terms that support/refute the claim).  
2. **Highlights factual inaccuracies** in the claim, such as incorrect origins, dates, or relationships, and links them to **direct contradictions** in the `passages` or `context`.  
3. **Mentions key terms or entities** critical to the claim (e.g., "Sly Stone," "Large Münsterländer," or "Hit Parader's 2006 ranking") to ensure summaries are actionable for query generation.  
4. **Balances accuracy** by acknowledging both correct and incorrect aspects of the claim, as seen in the examples (e.g., validating "The Wild Bunch" compilation while correcting the magazine's nationality and ranking year).  
5. **Avoids vague statements** like "no evidence" and instead specifies **which parts of the passages** are relevant (e.g., "Passage 6 confirms Hit Parader was American").  

**Generalizable Strategy**:  
- Always reference **specific passages** (e.g., "Passage 1 states...") to ground claims in the text.  
- Use **precise terminology** (e.g., "German Shorthaired Pointer" instead of "cousins") to align with the data.  
- Prioritize **clarity and specificity** to ensure summaries can guide further evidence retrieval.
```

### GEPA-MERGE — qwen3-8b

*(file: `hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 1432 chars

```text
Given the fields `claim` and `summary_1`, produce the field `query` by:  
1. **Analyzing the claim** to identify key entities, relationships, and factual assertions (e.g., names, organizations, locations, roles).  
2. **Cross-referencing the summary_1** to determine if it confirms, contradicts, or leaves unresolved aspects of the claim.  
3. **Formulating a query** that:  
   - Explicitly connects the claim and summary (e.g., "Is there evidence that X [from claim] aligns with Y [from summary]?").  
   - Includes **specific terms** from the claim and summary that are likely to appear in evidence (e.g., names, organizations, or technical terms).  
   - Prioritizes unresolved claims or contradictions (e.g., "Does the summary provide evidence to verify X?").  
4. **Avoiding vague phrasing** and ensuring the query is actionable for retrieving evidence (e.g., instead of "Clarify the route," use "What is the route of the railway that reopened Broadfield station?").  

**Key considerations**:  
- If the summary states the claim cannot be verified, the query should ask for evidence to resolve the uncertainty (e.g., "What evidence confirms or refutes X?").  
- If the summary contains specific terms (e.g., "Skittles," "Mars Incorporated") that are relevant to the claim, include them explicitly in the query.  
- Use domain-specific terminology (e.g., "boss," "chair," "manufacturer") as identified in the claim or summary.
```

**stage 2/4** — outputs `summary_2` — 1405 chars

```text
Given the fields `claim`, `summary_1`, and `summary_2`, generate the field `query` by formulating a question that:  
1. **Explicitly references specific terms** from the claim (e.g., names, titles, dates, or key facts) to ensure evidence retrieval.  
2. **Clarifies ambiguities** in the claim or summaries (e.g., conflicting details, missing evidence, or misattributions).  
3. **Focuses on the core discrepancy** between the claim and the summaries (e.g., "Is X accurate?" or "What evidence supports X?").  
4. **Includes all critical entities** mentioned in the claim and summaries (e.g., "John Candy," "Planes, Trains and Automobiles," "KLM," "OR Tambo International Airport") to guide evidence searches.  
5. **Avoids vague phrasing** and ensures the query is actionable for retrieving specific evidence (e.g., instead of "Is the claim true?" use "Does the evidence confirm John Candy's role in *Planes, Trains and Automobiles*?").  

**Key Considerations:**  
- If summaries mention missing evidence (e.g., a book title, a name, or a specific fact), the query must explicitly include these terms.  
- Prioritize questions that resolve contradictions or clarify incomplete claims (e.g., "Which entity holds the 18% stake in Comair?").  
- Use the structure: "Is [specific detail from the claim] accurate, given [specific evidence from summaries]?" or "What evidence supports [specific claim element]?"
```

**stage 3/4** — outputs `passages, summary` — 4192 chars

```text
**New Instruction for the Assistant**  

Given the fields `claim` and `passages`, produce the field `summary` that explicitly connects the claim's components to the evidence in the passages, ensuring all **key evidence** (e.g., names of individuals, specific events, domain-specific terms) is directly tied to the claim.  

**Task Description:**  
1. **Break Down the Claim:** Identify all assertions in the claim (e.g., specific facts, comparisons, or attributes), including **named entities** (e.g., people, titles, institutions) and **domain-specific terms** (e.g., "studio albums," "third-level colleges," "population statistics," "Grand Slam titles," "creative writing course directors").  
2. **Search for Evidence:** For each assertion, locate corresponding information in the provided passages. Use **exact matches** for named entities and **domain-specific terms** (e.g., "UEA Creative Writing Course Directors," "Grand Slam titles," "studio albums") to ensure precision. If a claim element is unverified, explicitly note the absence of evidence.  
3. **Verify Support:** Determine if the evidence supports, partially supports, or contradicts the claim. If evidence is missing, clarify that the claim cannot be fully verified.  
4. **Summarize Connections:** Explicitly link each part of the claim to the relevant evidence in the passages, including **all critical evidence** (e.g., "National Maritime College of Ireland," "Jimi Hendrix's career timeline," "1980 French Open – Mixed Doubles," "Lavinia Greenlaw"). Even if a passage does not directly address the claim, mention it if it contains **key evidence** related to the claim’s components.  
5. **Highlight Key Evidence:** Ensure the summary includes **all named individuals, specific events, and domain-specific terms** from the passages that directly or indirectly relate to the claim, even if they are not explicitly mentioned in the claim. This includes cross-referencing multiple passages to identify indirect connections (e.g., linking a person’s role to an institution or event).  

**Generalizable Strategy:**  
- **Prioritize exact matches** for named entities and domain-specific terms (e.g., "UEA Creative Writing Course Directors," "Grand Slam titles," "studio albums").  
- **Explicitly mention all key evidence** from the passages, even if it is not directly tied to the claim, to facilitate future retrieval.  
- **Clarify missing evidence** by stating that the claim cannot be fully verified due to absence of specific information.  
- **Use concise, factual language** to avoid ambiguity and ensure clarity in linking claims to evidence.  
- **Cross-reference multiple passages** to ensure no relevant details (e.g., "Lavinia Greenlaw," "1980 French Open – Mixed Doubles") are overlooked, even if they are not directly part of the claim.  

**Additional Focus Areas for Precision:**  
- **Clarify Misinterpretations:** If the claim misidentifies a term (e.g., conflating a wine with an opera), explicitly state the correct context (e.g., "Marzemino is a wine mentioned in *Don Giovanni*, not an opera").  
- **Contextual Associations:** Highlight implicit connections (e.g., "Chris Columbus’s association with American film production companies implies a likely American nationality").  
- **Unverified Components:** For claims with unverified elements (e.g., "Victor Slezak’s 1995 filmography does not include Patricia Arquette"), explicitly note the absence of evidence and its impact on the claim’s validity.  
- **Domain-Specific Terminology:** Ensure all domain-specific terms (e.g., "chamber opera," "third-level colleges") are explicitly tied to their definitions or examples in the passages.  

**Example of Key Evidence Inclusion:**  
- If the claim references "Johnny Depp," include evidence about his roles (e.g., *Ed Wood*) and related individuals (e.g., "Sarah Jessica Parker, Patricia Arquette").  
- If the claim involves a film title, cross-reference it with its director, cast, and production details (e.g., "Wonderstruck (2017) directed by Todd Haynes").  
- For indirect connections (e.g., "Lavinia Greenlaw’s role in a passage"), explicitly mention the relevance to the claim’s components.
```

**stage 4/4** — outputs `context` — 1374 chars

```text
Given the fields `claim`, `context`, and `passages`, produce the field `summary` that:  

1. **Explicitly connects each part of the claim to the relevant evidence in the passages**, using the exact terms from the claim (e.g., "Yes Minister," "Galleria Corporate Center") to ensure clarity.  
2. **Highlights key supporting or conflicting evidence** from the passages, citing specific passage numbers or identifiers (e.g., "Passage 1," "Passage 3") and quoting critical phrases (e.g., "film director," "headquartered in Baltimore").  
3. **Identifies missing evidence** relevant to the claim, even if not explicitly stated in the passages (e.g., "no evidence links Rodríguez to a Maryland-based publication" or "no mention of 'Galleria Corporate Center' in the passages").  
4. **Includes domain-specific details** (e.g., geographic locations, rankings, or specific titles) to ensure precision, as these are critical for query generation and fact-checking.  
5. **Maintains neutrality** by stating whether the claim is fully supported, partially supported, or unsupported, while clearly articulating the reasoning.  

**Generalizable Strategy**:  
- Always map each component of the claim to the most directly relevant passage(s).  
- Use precise terminology from the claim and passages to avoid ambiguity.  
- Explicitly note gaps in evidence to guide further investigation.
```

### Abl-SelectBestCandidate — qwen3-8b

*(file: `hoverBench_HoverMultiHop_Abl-SelectBestCandidate_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 2454 chars

```text
**Task Instruction: Generate a Precise, Evidence-Focused Query Based on Claim and Summary**  

**Objective:**  
Given the fields `claim` and `summary_1`, create a query that explicitly addresses both supported and unsupported elements of the claim by:  
1. **Identifying all key entities and relationships** (e.g., names, dates, locations, organizations, roles, and explicit connections) from the claim and summary.  
2. **Clarifying contradictions or ambiguities** in the summary (e.g., unverified assertions, missing evidence, or misattributed details).  
3. **Targeting specific evidence retrieval** by asking questions that directly reference entities, relationships, or facts in the provided passages (e.g., "Is there a passage that explicitly links [entity] to [fact]?" or "Which specific details in the text confirm [relationship] between [entity1] and [entity2]?").  
4. **Avoiding vague phrasing** by replacing general questions (e.g., "Is the claim true?") with precise queries that align with the summary’s analysis (e.g., "Does any passage confirm [specific detail] about [entity]?").  

**Key Strategies for the Query:**  
- **Explicitly list all named entities** (e.g., "Ettore Scola," "Jonathan Lynn," "Splatter Theatre," "Mick Napier") and relationships (e.g., "director of," "co-writer on") mentioned in the claim and summary.  
- **Highlight contradictions or gaps** in the summary (e.g., "The claim relies on unverified external information" → "Is there any passage that confirms [specific external detail] about [entity]?").  
- **Ensure the query connects entities to evidence** in the passages (e.g., "Does the text explicitly state that [entity] is associated with [fact]?" or "Which passage links [entity1] to [entity2] as described in the claim?").  
- **Use specific references** from the summary (e.g., "County Cork" in Example 2, "Yes Minister" in Example 3) to avoid missing critical evidence.  

**Example of Improved Query Structure:**  
If the summary states, "The claim is refuted because [X] is not mentioned in the passages," the query should ask:  
"Is there any passage that explicitly mentions [X] or links [X] to [related entity] as claimed?"  

**Note:** The query must ensure all entities and relationships from the claim and summary are directly addressed, with no omissions (e.g., "Splatter Theatre" in Example 1, "Yes Minister" in Example 3). Avoid general questions and focus on actionable evidence retrieval.
```

**stage 2/4** — outputs `summary_2` — 1632 chars

```text
Given the fields `claim`, `summary_1`, and `summary_2`, produce the field `query` that explicitly retrieves evidence relevant to the claim.  

**Key Requirements:**  
1. **Incorporate Key Terms:** Include specific names, titles, and relationships mentioned in the summaries (e.g., "Delmer Daves," "Jean-Pierre Jeunet," "Capitale de la douleur," "Moonrunners") to ensure the query targets the exact evidence discussed.  
2. **Address Missing Evidence:** Explicitly reference gaps in the summaries (e.g., "Delmer Daves' directorship," "Georges Bataille's nationality") to guide retrieval of unresolved details.  
3. **Avoid Assumptions:** Frame the query strictly based on the summaries, avoiding inferred connections not explicitly stated (e.g., "writer = director" or "actor = character").  
4. **Use Contextual Clues:** Leverage contextual details from the summaries (e.g., "Passages 2, 3, 5, 6, 7," "French Surrealist movement") to narrow the scope of evidence retrieval.  
5. **Clarify Ambiguities:** If the claim involves conflated roles (e.g., "character vs. actor," "show vs. film"), structure the query to disambiguate these relationships.  

**Example Strategy:**  
- For a claim about a person's role, ask: *"Is [Name] explicitly identified as [Role] in the provided passages?"*  
- For a claim involving connections between entities, ask: *"What evidence links [Entity A] and [Entity B] to [Relationship]?"*  
- For unresolved gaps, ask: *"Does the evidence confirm [Specific Detail] about [Subject]?"*  

This ensures the query directly addresses the claim's validity while aligning with the summaries' evidence and gaps.
```

**stage 3/4** — outputs `passages, summary` — 4874 chars

```text
Given the fields `claim` and `passages`, produce the field `summary` by evaluating the claim against the provided passages. Your summary must:  

1. **Clearly state whether the claim is supported, refuted, or partially supported** based on the evidence in the passages.  
   - Use explicit comparisons (e.g., "X is later than Y," "Z is a subset of W") to justify your conclusion.  
   - If the claim involves ambiguous terms (e.g., "Charpes Lane"), explicitly resolve ambiguities by cross-referencing with the passages (e.g., "Charles Lane" is explicitly mentioned in the passages).  

2. **Explicitly highlight the specific evidence from the passages** that directly relates to the claim:  
   - For **products** (e.g., "Skittles"), reference their manufacturer, release dates, or associated entities (e.g., "Mars Incorporated").  
   - For **breeds** (e.g., "German Longhaired Pointer"), connect to related breeds, development history, or functional roles (e.g., "Large Münsterländer").  
   - For **people/films** (e.g., "Victor Slezak," "Beyond Rangoon"), cross-check their appearances, roles, or production credits in the passages.  
   - For **ambiguous terms** (e.g., "Carlina"), clarify their direct or indirect connections (e.g., "Carlina acaulis" in the passage).  

3. **Identify all key terms or entities in the claim** and map them to relevant information in the passages:  
   - If a term (e.g., "David Bowman") is linked to a species but lacks classification details, note the absence of explicit evidence.  
   - If a term (e.g., "Australian Burn Gorman") is not explicitly mentioned in the passages, highlight the discrepancy between the claim’s assertion and the passage’s description (e.g., "English-American").  
   - For **misspelled or variant terms** (e.g., "Charpes Lane" vs. "Charles Lane"), explicitly note the potential typo and resolve it using the passage’s explicit mentions.  

4. **Note any missing evidence** from the passages that could strengthen or clarify the claim:  
   - For example, if the claim references a term not explicitly mentioned (e.g., "Australian Burn Gorman"), highlight the discrepancy and clarify the passage’s description (e.g., "English-American").  
   - If the claim relies on external knowledge (e.g., "Dieffenbachia" is a flowering plant), explicitly state that the passage does not provide this information.  
   - If the claim involves a **timeline or sequence** (e.g., "X occurred before Y"), use explicit dates or events from the passages to validate or refute this.  

5. **Avoid assumptions** beyond the provided passages:  
   - Do not infer relationships (e.g., "X is a sequel to Y") unless explicitly stated.  
   - If the claim requires external knowledge (e.g., "Buddleja davidii" is native to China), clarify that the passage does not confirm this.  

**Domain-Specific Reasoning Examples:**  
- **Products**: If the claim mentions a product (e.g., "Skittles"), reference its manufacturer, release date, or related entities (e.g., "Mars Incorporated").  
- **Breeds**: If the claim references a breed (e.g., "German Longhaired Pointer"), connect to related breeds (e.g., "German Shorthaired Pointer") or historical development (e.g., "Large Münsterländer").  
- **People/Films**: If the claim involves a person or film (e.g., "Victor Slezak," "Beyond Rangoon"), cross-check their appearances, roles, or production credits in the passages.  
- **Ambiguity Resolution**: If a term (e.g., "Carlina") could refer to multiple entities, explicitly tie it to the specific passage evidence (e.g., "Carlina acaulis" in the passage).  

**Critical Connections to Emphasize:**  
- **Explicit vs. Implicit Evidence**: Distinguish between direct statements (e.g., "X is a sequel to Y") and inferred relationships (e.g., "X likely influenced Y").  
- **Ambiguity Resolution**: If a term (e.g., "Carlina") could refer to multiple entities, explicitly tie it to the specific passage evidence (e.g., "Carlina acaulis" in the passage).  
- **Missing Evidence**: If the claim references a term not in the passages (e.g., "Australian Burn Gorman"), note the discrepancy and clarify the passage’s description (e.g., "English-American").  

**Additional Guidelines for Precision:**  
- **Map Key Terms**: Ensure all terms in the claim (e.g., "Lucille Ball," "Charpes Lane") are explicitly tied to the passages, even if they require resolving typos or indirect connections.  
- **Highlight Conflicts**: If the claim contradicts the passages (e.g., "X is a subset of Y" vs. "Y is a subset of X"), explicitly state the conflict.  
- **Use Passage Citations**: Always reference specific passages (e.g., "Passage [1]") to support claims about timelines, roles, or relationships.  

Ensure your summary is concise but includes all critical connections between the claim and the evidence, enabling further queries or verification.
```

**stage 4/4** — outputs `context` — 1972 chars

```text
Given the fields `claim`, `context`, and `passages`, produce the field `summary` that:  
1. **Evaluates the claim's support** (fully supported, partially supported, or unsupported) based on the provided evidence.  
2. **Explicitly connects the claim to relevant passages** by citing specific details (e.g., names, events, terms) from the `passages` that support, contradict, or lack evidence for the claim.  
3. **Highlights missing evidence** that would be required to fully verify the claim, including terms or entities mentioned in the `claim` but absent from the `passages`.  
4. **Clarifies ambiguities** in the claim (e.g., undefined terms, assumptions) that the `passages` do not resolve.  
5. **Uses precise language** to avoid vague statements, ensuring the summary directly links the claim's components to the evidence.  

**Key Considerations for the Summary:**  
- If the claim references specific entities (e.g., people, events, awards), explicitly state whether those entities are mentioned in the `passages` and how they relate to the claim.  
- If the claim assumes a comparison (e.g., "X has more awards than Y"), ensure the summary notes whether the `passages` provide data for both X and Y.  
- If the `passages` mention terms (e.g., "bass," "countertenor") that are relevant to the claim, explain their significance in the context of the claim.  
- If the `passages` lack direct evidence for a key part of the claim, explicitly state what is missing (e.g., "No passage mentions [Entity X]’s awards").  
- Avoid assumptions beyond the `passages`—only use information explicitly provided.  

**Example Structure for Summary:**  
"The claim is [supported/partially supported/unsupported] because [specific reasoning]. Key evidence includes [cited passage(s)] and [specific detail]. Missing evidence includes [unmentioned term/entity], which would be required to fully verify [specific part of the claim]. Ambiguities include [unclear term or assumption]."
```

### MIPROv2-Heavy — qwen3-8b

*(file: `hoverBench_HoverMultiHop_MIPROv2-Heavy_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 469 chars

```text
In a high-stakes scenario where the accuracy of your query directly determines the validity of a complex claim, generate a precise query that builds on the provided claim and summary to uncover critical evidence. Use step-by-step reasoning to identify gaps in the initial summary, then craft a query that explicitly targets these gaps to retrieve additional supporting information. Ensure your query is unambiguous and directly addresses the claim's unresolved aspects.
```

**stage 2/4** — outputs `summary_2` — 375 chars

```text
You are a fact-checking assistant specializing in multi-hop reasoning and information synthesis. Given the fields `claim`, `summary_1`, and `summary_2`, generate a precise query that synthesizes the original claim with the two summaries to probe for deeper contextual relationships, resolving ambiguities and confirming supporting details through targeted evidence retrieval.
```

**stage 3/4** — outputs `passages, summary` — 265 chars

```text
Given the fields `claim` and `passages`, generate a structured reasoning process that analyzes the claim step-by-step using the provided evidence, and produce a concise summary that distills the key findings and evaluates the claim's validity based on the evidence.
```

**stage 4/4** — outputs `context` — 579 chars

```text
Given the fields `claim`, `context`, and `passages`, perform multi-hop reasoning to generate a structured summary that validates or refutes the claim. First, analyze the claim and contextual information to identify key relationships. Next, evaluate evidence from the retrieved passages to build a coherent narrative. Construct a logical reasoning chain connecting the claim to supporting or contradictory evidence. Finally, produce a concise summary that explicitly confirms or contradicts the claim based on your analysis, citing relevant evidence from the context and passages.
```

### GRPO — qwen3-8b

*(file: `hoverBench_HoverMultiHop_GRPO_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 66 chars

```text
Given the fields `claim`, `summary_1`, produce the fields `query`.
```

**stage 2/4** — outputs `summary_2` — 79 chars

```text
Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.
```

**stage 3/4** — outputs `passages, summary` — 67 chars

```text
Given the fields `claim`, `passages`, produce the fields `summary`.
```

**stage 4/4** — outputs `context` — 78 chars

```text
Given the fields `claim`, `context`, `passages`, produce the fields `summary`.
```

---

## HotpotQA (multi-hop)


### GEPA — gpt-41-mini

*(file: `HotpotQABench_HotpotMultiHop_GEPA_gpt-41-mini.pkl`)*


**stage 1/3** — outputs `summary_1, reasoning, query` — 3840 chars

```text
You will be given two input fields: `question` and `summary_1`.

Your task is to generate a new search query (`query`) optimized for the **second hop** of a multi-hop retrieval system. The original user question is typically complex and requires information from multiple documents to answer. The first hop query is the original question used to retrieve an initial set of documents. Your goal is to generate a **second hop query** that retrieves *additional relevant documents* that were *not* found in the first hop but are necessary to answer the original question completely.

Detailed task instructions and hints:

1. **Input Understanding:**
   - `question` is the original multi-hop question posed by the user.
   - `summary_1` is a concise summary of information from a document retrieved in the first hop, which partially addresses the question.

2. **Purpose and Context:**
   - Your generated `query` aims to find the *missing pieces* of information needed to fully answer the `question`.
   - The multi-hop retrieval system works in stages:
     - First hop: The original question returns some documents.
     - Second hop: Your query must help retrieve any *other relevant documents* NOT found in the first hop that hold complementary or broader context necessary for final answer extraction.

3. **Key Observations from Examples and Feedback:**
   - First-hop documents often cover one entity or aspect in the question.
   - Remaining relevant documents often involve connected or higher-level concepts mentioned in `summary_1` but not explicitly asked in the original question.
   - The `query` should be formulated to explicitly target these *missing*, but logically linked, documents.
   - Avoid merely paraphrasing the original question or restating known facts from `summary_1`.
   - Instead, infer what broader or related entities/concepts might provide the crucial missing information.
   - For example, if `summary_1` describes a population for a small civil parish, but the question wants total population of the wider region, your `query` should target that wider region (e.g., "Madeira archipelago population in 2011").
   - Similarly, if `summary_1` covers a song and the question wants the album it came from, but first hop got song-level documents, your query should retrieve documents about the album itself.

4. **How to Build the Query:**
   - Identify the entities or topics mentioned in `summary_1` that appear related but different from first-hop documents.
   - Reframe the query to explicitly mention these broader or related entities connected to the original question.
   - Include relevant key context from the question to maintain specificity, but shift focus to the missing piece.
   - The goal is to retrieve documents that link or complement what was retrieved initially.

5. **Practical Strategy:**
   - Read the `summary_1` carefully to spot references to bigger contexts or other entities not covered in the first hop.
   - Ask yourself, "What entity or aspect does this summary hint at that could answer the original question but was not found yet?"
   - Formulate a precise, focused factual query targeting that entity or concept to retrieve the missing documents.

6. **Output:**
   - Produce only the field `query` as a clear, concise question or keyword phrase designed for efficient retrieval of **second-hop documents**.
   - Ensure the query relates logically to the original question while targeting the broader or complementary knowledge identified in `summary_1`.
   - Do **not** include the original question or simply rephrase it.
   - Do **not** duplicate information already well-covered by the first hop retrieval.

By following these principles, you will help the multi-hop retrieval system find all necessary documents to answer the multi-faceted original question completely.
```

**stage 2/3** — outputs `summary_2, answer` — 4207 chars

```text
Task Description:

You are given three fields as input: `question`, `summary_1`, and `summary_2`. Your goal is to produce an `answer` field that directly and explicitly responds to the question using the information from the two summaries, enhanced by your authoritative domain knowledge when needed.

Input Format:

- `question`: A natural language question that may require a fact, definition, name, date, yes/no response, or other specific information.
- `summary_1` and `summary_2`: Two independently generated summaries or snippets containing information related to the question. They may vary in completeness, accuracy, and specificity.

Requirements and Approach:

1. **Understand the question precisely.** Determine exactly what is asked—whether a name, a specific fact, a date, or a yes/no answer.

2. **Compare both summaries.** Analyze the content of `summary_1` and `summary_2`:
   - If they agree and directly answer the question, use this as primary evidence.
   - If one summary provides a fact that the other does not mention, carefully evaluate its plausibility.
   - If the summaries conflict, use domain expertise and authoritative knowledge to resolve or explicitly state uncertainty.

3. **Domain-specific factual verification and nuance:**
   - **Names and nicknames:** Provide only the specific nickname or name when asked, without extra phrasing. For example, when asked for the nickname of a person or entity, respond with the nickname alone, not a full sentence.
   - **Nationality and identity distinctions:** Use the most precise terms aligned with factual correctness and common usage (e.g., “English” vs. “British”) based on domain knowledge.
   - **Dates and historical facts:** Verify dates or historical claims with domain knowledge to pick the correct fact, especially when there might be confusion between franchise start dates vs. event dates etc.
   - **Yes/no questions:** Prefer concise answers of “yes” or “no” only, unless the question demands elaboration.
   - **Types or categories:** If a question asks about the type or category (e.g., type of company), provide the most direct concise phrase without including adjectives like nationality unless asked explicitly.

4. **Answer conciseness and relevance:**
   - Provide a brief and direct answer to the question.
   - Avoid repeating or restating the question.
   - Avoid unnecessary context unless requested or needed for clarity.
   - Avoid constructing full sentences unless needed; for example, answers to nickname or yes/no questions should be as short and specific as possible.

5. **When authoritative knowledge supplements the summaries:**
   - If the summaries are incomplete or potentially inaccurate, incorporate trusted knowledge from your training about the topic to provide the correct and precise answer.
   - For example, when a summary gives a year that conflicts with known release dates or factual details, prefer the verified date.
   - When the summaries differ in style (one uses a formal phrase, another provides just the nickname), respond with the correct, clean answer format (e.g., just the nickname alone).

6. **Examples of correct reasoning and answers:**
   - Question: “What is the nickname of the 2005 Toyota Grand Prix of Long Beach Polesitter ?”
     - Correct answer: `the thrill from West Hill`
   - Question: “What type of company is Zipcar led by Scott Griffith from 2003-2013?”
     - Correct answer: `car-sharing company`
   - Question: “Who was the partner of British comic book artist, Henry Flint, that helped create Zombo?”
     - Correct answer: `Al Ewing`

Summary:

- Use both summaries as primary but not sole evidence.
- Reliably verify and contextualize facts using domain knowledge, especially for nationality, dates, nicknames, company types, and yes/no questions.
- Provide short, direct answers matching the specificity requested.
- Avoid unnecessary elaboration unless explicitly required.
- Explicitly resolve conflicts or ambiguity using your knowledge or state uncertainty when appropriate.

This approach ensures that answers are both accurate and concise, suitable for direct consumption or integration in knowledge bases or question-answering systems.
```

**stage 3/3** — outputs `passages, summary` — 2430 chars

```text
 
You are a first-hop **summarization module** in a multi-hop question answering (QA) system. Your task is to generate a concise, informative `summary` given two input fields: a `question` and a list of relevant `passages`.

Your goal is to extract and synthesize key information from the retrieved passages that:

1. Directly relates to the initial question.
2. Captures the core facts and entities needed to understand the scope and context of the question.
3. Includes relevant connections, bridging entities, dates, locations, or descriptions that enable the system to devise focused and effective follow-up queries in subsequent hops.
4. Provides a strong factual foundation for downstream answer generation modules.

**Task specifics and best practices:**

- The `summary` must represent a distilled synthesis, not just a compression or extractive snippet.
- Explicitly include cited passage titles or key entity labels (e.g., "Children in Need 2006 | ..." or "Anthony Levandowski | ...") in your summary to highlight the origin of information.
- Incorporate sufficient context to hint at missing or un-retrieved supporting facts, thus enhancing the multi-hop retrieval process.
- When the question asks for an attribute (e.g., nationality, location, company origin), ensure you provide:
   - Identification of the relevant subject or entity mentioned in the passages.
   - The extracted attribute or relevant information as stated or implied.
   - Bridging details that could help the system pursue remaining information in the next retrieval step.
- Avoid forming a final answer; instead, focus on "what is known now" from the input documents to facilitate further query refinement.

**Examples of critical elements to include:**

- Entity names, roles, titles, and dates tied to the question.
- Names of organizations or locations connected through intermediary entities.
- Distinctive identifiers or clarifications that can help narrow down next-step retrieval (such as "Natasha Kaplinsky is an English presenter," "Kapolei is a city on Oahu," or "Waymo spun out of Alphabet").

**Format of output:**

Provide a paragraph or a few sentences that cohesively summarize the key passages in relation to the question, referencing passage titles or entities to frame facts clearly.

---

This approach ensures the summary is both informative for next-hop retrieval and foundational for final answer extraction in multi-hop QA.
```

### GEPA-MERGE — gpt-41-mini

*(file: `HotpotQABench_HotpotMultiHop_GEPA-MERGE_gpt-41-mini.pkl`)*


**stage 1/3** — outputs `summary_1, reasoning, query` — 3326 chars

```text
Task: Given a natural language `question` and a `summary_1` that contains partial information relevant to answering the question, generate an optimized `query` to be used as the **second hop query** in a multi-hop retrieval system.

Context:
- The retrieval system works in multiple hops:
  - **First hop:** Uses the original question to retrieve an initial set of documents.
  - **Second hop:** Your generated `query` is used to retrieve additional relevant documents **that were not retrieved in the first hop**, crucial for answering the question fully.

Input:
- `question`: The original complex question posed by the user.
- `summary_1`: A summary of information extracted from documents retrieved in the first hop related to the question. This summary partially addresses the question but may miss some entities or facts necessary for the final answer.

Output:
- `query`: A refined and targeted natural language query designed specifically to:
  1. Identify and retrieve **missing relevant documents** or entities not found in the first hop.
  2. Connect known facts from `summary_1` with gaps or alternative angles needed to complete the answer.
  3. Target documents related to entities or attributes that have been overlooked or underrepresented in the first-hop retrieval.

Guidelines and Important Details:
- Your goal is **not** to re-query what was already retrieved, but to help uncover the **remaining necessary documents** or information.
- Analyze what documents or entities were covered in the first hop (as hinted by `summary_1`) and identify which relevant ones are missing.
- Use clues and connections extracted from `summary_1` to formulate the query. For example:
  - If part of the answer involves comparing two entities and one was already retrieved, the query should focus on retrieving information about the other entity.
  - If the summary confirms some facts but leaves another related entity or property ambiguous, craft a query that explicitly targets those ambiguous or missing aspects.
- Avoid broad or generic queries that merely rephrase the original question. Instead, be as specific as possible about what is missing or unknown after the first-hop retrieval.
- Consider possible alternative relations or linked topics that can bridge the information gap.
- You may ask about related entities, events, locations, or attributes that complement the known facts.
- The generated query should be concise yet sufficiently descriptive to retrieve the missing entity documents.
- Use natural language questions or phrases that are likely to surface relevant documents in a retrieval system.

Examples of strategy:
- When a question contrasts two entities and the summary confirms facts about only one, generate a query focusing on retrieving information about the other entity.
- When the summary states a negative or limiting fact (e.g., a stadium did not host certain teams), the query should explore other possibilities or related details not yet covered.
- When the original question includes multi-part dependencies, your query should isolate the missing link or fact necessary to complete the chain.

By following these principles, the queries you generate will improve the multi-hop retrieval system’s ability to gather all relevant documents needed to answer complex questions accurately.
```

**stage 2/3** — outputs `summary_2, answer` — 4343 chars

```text
Task Description:

You will be given three input fields:  
- `question`: a natural language question specifically requesting a concise factual answer. The question will typically ask for a specific entity such as a person's name, a date, a place, a title, or a simple fact.  
- `summary_1`: a textual summary containing relevant facts, possibly with detailed context, that could answer the question.  
- `summary_2`: a second textual summary with overlapping or complementary relevant facts that also potentially answer the question.

Your objective is to produce a single output field:  
- `answer`: a concise, minimal, and direct factual answer extracted and inferred from the provided summaries.

Instructions and Best Practices:

1. **Precisely identify what the question asks for**  
   Determine the exact factual entity or value requested (e.g., name of person, title of movie, country, year, band name).

2. **Synthesize both summaries to validate and extract facts**  
   - Compare `summary_1` and `summary_2` for consistency on key facts relevant to the question.  
   - Use the summaries in combination to confirm or fill any missing details.  
   - If both summaries agree or nearly agree, trust the shared information.  

3. **Provide the exact minimal answer only**  
   - Supply only the essential answer without restating the question or adding explanatory context.  
   - For example, if the question asks for a person's name, provide the name alone (e.g., "Andre Dirrell").  
   - Do not include relational or descriptive words unless explicitly requested (e.g., do not say "Anthony Dirrell is the brother of Andre Dirrell" but simply "Andre Dirrell" if asked for the brother's name).  
   - In cases where the question expects a title, year, or place, provide just that fact.  
   - For choice questions (e.g., which band or film), supply only the correct choice name.  

4. **Ensure factual accuracy and clarity**  
   - The answer must be based solely on information found in the summaries.  
   - Avoid including personal opinions, unrelated facts, or extra details not directly answering the question.  

5. **Domain-specific considerations:**  
   - **Boxing domain:**  
     Understand that titles like “IBF interim super middleweight title” or “WBC super middleweight title” refer to professional boxing championships. Names often include birthdates or career details, which help verify identities but are usually not part of the answer unless asked. Relationships (e.g., brothers) are key but supply only the entity name when prompted.  
   
   - **Music domain:**  
     Bands’ formation years, member roles, and member names distinguish groups. Provide just the band or member name as the answer. Roles or details (like “lead vocalist”) are supplied only if the question directs you.  

   - **Entertainment domain:**  
     Years of birth, album titles, movie names, or director names identify time-based or identity-related answers. Provide just the year, name, or title as requested.  

6. **Reasoning and explanation should be optional and separate from your final output**  
   - When reasoning about your answer, you may highlight where both summaries agree or support the answer, mention confirming dates or facts, and summarize how you synthesized the information.  
   - The final output must only be the concise, factual answer field without reasoning or extra commentary.

7. **Common pitfalls and corrections:**  
   - Do not confuse the entity sought by the question with related entities mentioned in summaries (e.g., if the question asks for a movie title, do not answer with an actor’s name even if both appear).  
   - If the question requests a movie or work title associated with an actor, answer with the title alone.  
   - Avoid quoting full sentences, pronouns, or relational phrases unless explicitly requested.

Input / Output Format:

Inputs:  
- `question` (string) — the factual question.  
- `summary_1` (string) — first relevant textual summary.  
- `summary_2` (string) — second relevant textual summary.

Output:  
- `answer` (string) — the minimal, concise factual answer strictly drawn from the summaries.

This approach ensures consistent, accurate extraction of direct factual answers from provided summaries across diverse domains like boxing, music, and general entertainment.
```

**stage 3/3** — outputs `passages, summary` — 2799 chars

```text
You are given two input fields: `question` and `passages`. Your task is to generate a concise, well-structured `summary` based on the provided passages that directly addresses the question.

This summary serves as the first-hop summarization in a multi-hop QA system and must accomplish two key objectives:

1. **Facilitate effective follow-up querying:** Include crucial entities, their relationships, and bridging facts that might currently be missing but are needed to generate precise follow-up queries in subsequent hops.

2. **Provide a solid foundation for final answer generation:** Capture definitive, relevant facts that are clearly supported by the retrieved passages and help downstream components arrive at the correct answer.

**Detailed Guidance:**

- **Focus on synthesis over compression:** Don’t merely shorten the information. Instead, integrate relevant facts from all the passages to form a coherent, informative, and focused summary.

- **Include key entities with short factual context:** When passages mention people, organizations, shows, or other entities critical to answering the question, include their key attributes (e.g., titles, roles, dates, relationships) that are explicit or strongly implied in the passages.

- **Signal missing but necessary information:** If the evidence hints at connections or facts not fully covered by the current passages but required to resolve the question, highlight them indirectly by including related entities or possible bridging concepts. This helps ensure follow-up retrieval targets the right knowledge gaps.

- **Use exact or near-verbatim factual representations:** Summaries should contain accurate details like full names, dates, titles, roles, and relationships drawn faithfully from the passages.

- **Prefer clarity and precision:** The summary should make the key facts and their relevance to the question unambiguous, enabling later modules to focus on missing details efficiently.

- **Avoid speculation:** Do not inject external knowledge or guess answers if the passages do not provide them; instead, represent the known facts and highlight what is missing.

- **Maintain logical structure:** Organize the summary so that each important entity or concept is introduced with context that reveals its connection to the question.

**Example format elements:**  
- Entity Name | Brief description with relevant attributes.  
- Relationships or roles clearly linked to the question.  
- Dates, titles, or other factual data related to the question’s core concepts.

**In essence:** Your summary is a synthesized distillation of retrieved evidence that bridges the gap toward a full answer by encapsulating all relevant factual clues and signaling gaps, thereby enabling effective subsequent retrieval and reasoning steps.
```

### Abl-SelectBestCandidate — gpt-41-mini

*(file: `HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_gpt-41-mini.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 3732 chars

```text
Task Description:
You are tasked with generating optimized search queries for the **second hop** of a multi-hop retrieval system. The system's goal is to answer a complex question by retrieving relevant documents step-by-step: 
- The **first hop** uses the original question to retrieve a set of documents. 
- The **second hop** uses a derived query based on the initial retrieved content (summarized in `summary_1`) and the original question (`question`) to locate any remaining relevant documents necessary to fully answer the original question.

Your inputs are two textual fields:  
- `question`: The original multi-hop question requiring retrieval of multiple linked facts/documents to answer completely.  
- `summary_1`: A summary of information obtained from the first hop retrieval, typically containing partial answers or contextual details relevant to the question.

Your output is a single field:  
- `query`: An optimized search query designed to retrieve additional relevant documents that were not found in the first hop.

Key Domain-Specific Considerations and Strategies:  
1. **Second-Hop Optimization:**  
   - Your query must explicitly leverage information in `summary_1` combined with unresolved or ambiguous elements in the `question`.  
   - The goal is to retrieve missing or complementary documents necessary to fully answer the original question, not just reiterate or confirm information already obtained.

2. **Multi-Hop Reasoning:**  
   - Analyze the entities, events, or relationships mentioned in both the `question` and `summary_1`.  
   - Identify which components of the original question are confirmed, which are uncertain, and what supporting knowledge documents remain outstanding.  
   - Formulate the query to bridge the gap indicated by the missing or uncertain information.

3. **Targeting Missing Documents:**  
   - Review the facts confirmed in `summary_1` and exclude them from the query focus unless needed as context.  
   - Identify documents still needed (e.g., a person’s biography, a film’s production details, a political figure’s policy proposals) and craft the query to retrieve those explicitly.  
   - Use named entities and key concepts from `summary_1` to connect to potentially missing documents.

4. **Avoid Overly Redundant or Broad Queries:**  
   - Do not just rephrase the original question or ask for the entire answer again.  
   - The query should efficiently target the next retrieval step by focusing on the unresolved link in the multi-hop chain.

5. **Example:**  
   - If the first hop summary identifies the film matching most criteria but leaves the cast unclear, the second-hop query should explicitly ask about the cast or actor-film connection to uncover missing cast documents.  
   - If the summary confirms a campaign relates to proposed cuts but lacks details about the proposer, the second-hop query should target the political figure or policy to retrieve those related documents.

6. **Generalizable Strategy:**  
   - Use a reasoning step (optional in your output but helpful internally) to clarify what is known, what remains unknown, and what document types are missing.  
   - From that reasoning, compose a focused, precise query emphasizing the specific missing fact(s) or entity relations linking known facts to the final answer.

In summary, your role is to synthesize a succinct retrieval query that:  
- **Utilizes information from the original question and first-hop summary,**  
- **Targets information gaps or unresolved parts of the original multi-hop question,**  
- **Enables the multi-hop retrieval system to locate all remaining relevant documents,**  
- **Facilitates answering the original question completely and accurately.**
```

**stage 2/4** — outputs `summary_2, answer` — 3387 chars

```text
You will be given three fields as input: `question`, `summary_1`, and `summary_2`. Your task is to produce the field `answer`.

Task objective:
- Provide a concise, precise, factual answer to the question using only the combined explicit information from `summary_1` and `summary_2`.
- Use only explicit facts stated in the summaries. If there is conflicting or incomplete information, carefully reconcile it to produce the most accurate and unambiguous answer possible.
- If information is missing or insufficient to answer definitively, respond with a clear negative fact when appropriate (e.g., "No"), rather than a vague or noncommittal answer like "It cannot be determined," if the facts imply a negative outcome.
- Favor minimal direct answers such as entity names, titles, league names, or clearly stated facts, avoiding verbose explanations.
- Do not restate or copy entire sentences unless necessary for factual precision or clarity.

Domain-specific considerations:
- People and film industry: Provide full, recognized professional names only (no nicknames, partial names). Do not mention any film or show titles unless explicitly asked.
- Titles of works (films, TV shows, books): Provide the officially recognized or canonical title only, without qualifiers or extra explanation.
- Sports teams and leagues: Provide the precise league or series name (e.g., "Serie B") only.
- Origins of phrases or terms linked to works: Provide just the source name (film or show name) succinctly.
- Use standard, common, or official proper nouns exactly as presented.
- For questions about whether two persons share the same citizenship or similar categorical attributes and one person’s info is missing yet the other is known, answer clearly "No" if the missing info and known info imply difference, rather than an uncertain or inconclusive reply.
- If the facts imply a negative or no, answer accordingly rather than making vague statements such as "cannot be determined," unless facts are truly insufficient.

General strategy:
1. Analyze both summaries closely to identify explicit and consistent facts relevant to the question.
2. Compare details to resolve discrepancies in favor of the most explicit, direct, and relevant information.
3. Extract only the explicit fact(s) answering the question, no reasoning or process explanation.
4. If one summary gives a clear fact that the other omits, use the fact as reliable.
5. When information is incomplete but sufficient to answer negatively or definitively, prefer a definitive short answer over uncertainty.
6. Avoid including additional context, background information, or explanations. Provide minimal, stand-alone factual answers.

Answer format:
- One or two words or a short phrase directly answering the question.
- Use exact proper names or titles as relevant.
- Never provide combined or mixed answers (e.g., no actor plus film title unless asked).
- Do not use phrases like “it cannot be determined” unless absolutely no relevant fact is available.
- If the answer is negative or the two entities differ on a categorical fact, say "No" directly.

This instruction is designed specifically for fact-based QA tasks involving knowledge about people (especially in film/music/sports), works of art, sports teams, and entity attributes, where precision and conciseness are paramount, and minimal direct fact responses are required.
```

**stage 3/4** — outputs `passages, summary` — 5041 chars

```text
You are given two inputs:

- `question`: a natural language question, often requiring multi-hop reasoning to answer, meaning it may require synthesizing information from multiple knowledge sources or documents.
- `passages`: a list of text passages retrieved as evidence relevant to the question, typically from the first retrieval hop in a multi-hop question answering pipeline.

Your task is to produce a **concise, factual, and well-integrated summary** that distills the most critical information from the provided passages specifically relevant to answering or narrowing the question. This summary serves two critical functions:
1. To support downstream modules in generating focused, effective follow-up queries for subsequent retrieval hops.
2. To provide a strong foundation for final answer synthesis by clearly mapping out known facts and suggesting promising leads for further expansion.

---

### Key Guidelines for Your Summary:

1. **Extract and emphasize essential factual content directly tied to the question.**  
   This includes, but is not limited to:
   - Full names and aliases of people, places, organizations.
   - Dates (birth, death, event dates).
   - Locations (city, county, state, country).
   - Formal titles, roles, affiliations, occupations.
   - Events, relationships, collaborations, and key entities/mechanisms.
   - Unique identifiers or attributes that can help disambiguation (e.g., IATA codes for airports, official titles).

2. **Integrate information across multiple passages into a cohesive, logically connected narrative.**  
   Avoid listing or paraphrasing passages independently. Instead, weave facts together smoothly to reduce ambiguity and overlap.

3. **Explicitly highlight “bridging facts” or clues that link across documents or concepts for follow-up queries.**  
   These can be:
   - Key individuals and their relationships or collaborative partners.
   - Shared organizations, project titles, or event names.
   - Distinctive qualifiers that differentiate similarly named entities.
   - Specific dates or events that can anchor next-hop retrieval.

4. **Make the summary self-contained and organized,** so downstream modules can use it without needing external context.  
   Structure information using the format:  
   `Entity or Document Title | Synthesized key facts relevant to the question.`  
   This enables traceability and clarity.

5. **Cover all relevant passages and emphasize those containing pivotal or linking information.**  
   Avoid unrelated or extraneous details, unless they clearly clarify or connect the core facts.

6. **Balance detail with conciseness.**  
   Include sufficient specifics for unique identification and relationship mapping but avoid verbose repetition or trivia.

7. **Avoid directly compressing or paraphrasing passages; focus on synthesis.**  
   Combine insights from several sources into a unified factual snapshot.

---

### Additional Expert Insights Gleaned from Examples:

- When retrieved passages reference entities related to the question but omit a key fact (e.g., a tag team name, a film’s release year), use the summary to highlight missing yet closely connected concepts or entities to prompt targeted follow-ups (e.g., mention “The Mega Powers” as the tag team involving Hulk Hogan, or “Cars (film)” as the likely movie associated with Sally Carrera).

- Always represent connections that bridge to missing information, enabling the system to generate follow-up queries that surface unreturned but crucial documents.

- If multiple passages discuss overlapping subjects (e.g., different members of the same collective or family), integrate these details to form a comprehensive view, emphasizing relationships and relevant roles.

- Use proper names, dates, and precise identifiers to minimize ambiguity, especially when entities have common names or there are closely related subjects.

- Summaries should be designed as “bridges” between sparse, raw retrieved snippets and sophisticated, multi-hop query generation or final answer modules.

---

### Recommended Approach/Process:

1. Carefully read all passages relative to the question.
2. Identify all named entities, dates, relationships, roles, and events that directly inform or help narrow the question.
3. Pinpoint bridging clues — unique names, titles, roles, or relationships — likely to drive subsequent retrieval stages.
4. Synthesize and integrate facts into concise but complete paragraphs or bullet points labeled by entity or document title.
5. Avoid redundancy or inclusion of irrelevant facts.
6. Maintain traceability of facts to original documents for transparency.
7. Ensure your summary provides both direct answer clues and hints about missing or unexplored knowledge areas.

---

Following these instructions ensures your summary serves as a rich, precise, and strategically designed knowledge snapshot that powers robust multi-hop question answering pipelines — facilitating both targeted next-step retrieval and high-quality final answer generation.
```

**stage 4/4** — outputs `context` — 3260 chars

```text
You are a summary generation module within a multi-hop question answering system. Your task is to produce a concise, well-structured, and informative summary by synthesizing and integrating information from three inputs: a question, an intermediate summary (called "context"), and a set of newly retrieved passages ("passages"). 

The goal of your summary is to aggregate all relevant facts and connections from the inputs that enable the next answer generation module— which only sees your summary and not the original inputs—to accurately and confidently answer the multi-hop question. Therefore, your summary must include all key supporting information and explicitly link pertinent entities, events, dates, and roles mentioned in the question and context, leveraging detailed evidence from the passages.

Key details and requirements for the task:

1. **Input Format**:
   - **question**: a multi-hop (multi-step reasoning) question requiring synthesis across different facts, entities, or events.
   - **context**: an intermediate summary or synthesis already partially addressing the question; your summary should build on, clarify, or expand this.
   - **passages**: newly retrieved textual snippets containing additional factual information relevant to the question.

2. **Output**:
   - **summary**: a precise synthesis integrating all relevant information from the context and passages, connecting them explicitly to the question.
   - It must be self-contained and sufficient for a separate answer generation module to produce the final answer without access to original inputs.

3. **Factual and Domain-Specific Knowledge Examples** (illustrative):
   - Historical figures and events, e.g., Johann Tserclaes, Count of Tilly commanding Catholic League forces in the Thirty Years' War, commanding at the Battle of Mingolsheim against General von Mansfeld.
   - Nationalities and professions, as in tennis players Robert Lindstedt (Swedish) vs Gail Chanfreau (French).
   - Biographical and career transitions, e.g., Luke Goss, former drummer of 1980s band Bros who later acted in the 2002 film Blade II based on Marvel Comics character Blade.
 
4. **Generalizable Strategy**:
   - Carefully identify and extract all entities, events, dates, roles, and relationships explicitly or implicitly linking the question, the context, and the passages.
   - Bridge gaps by inference if some links are not directly stated but strongly implied by the input materials.
   - Avoid including irrelevant information that does not contribute to answering the question.
   - Ensure the summary is comprehensive enough to support the expected final answer confidently.
   - Maintain clarity and coherence, using full descriptive phrases (e.g., full names and titles) and specifying dates or places when relevant.

5. **Avoid**:
   - Leaving out key facts that connect entities or events critical for answering the question.
   - Redundancy without added clarity.
   - Vague or incomplete references that would prevent the answer module from deducing the final answer.

By adhering to these principles, your summary will robustly support multi-hop question answering that demands synthesizing information from multiple sources and bridging knowledge gaps intelligently.
```

### MIPROv2-Heavy — gpt-41-mini

*(file: `HotpotQABench_HotpotMultiHop_MIPROv2-Heavy_gpt-41-mini.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 601 chars

```text
You are an expert multi-hop question answering system designed to refine retrieval queries for complex questions. Given the original question and an initial summary of retrieved documents (summary_1), think step by step to analyze how the summary relates to the question. Use this reasoning to generate a focused and precise query that will guide the retrieval of additional relevant information needed to answer the question completely. Your output should include a clear chain-of-thought reasoning process followed by the refined query that targets complementary evidence for the next retrieval hop.
```

**stage 2/4** — outputs `summary_2, answer` — 412 chars

```text
You are an expert multi-hop reasoning assistant skilled in synthesizing information from multiple summaries to answer complex questions. Given the `question`, along with two intermediate summaries `summary_1` and `summary_2` that contain relevant evidence, carefully analyze and integrate the information step-by-step to produce a clear, logical reasoning process followed by a concise and accurate final answer.
```

**stage 3/4** — outputs `passages, summary` — 347 chars

```text
Given a question and a set of related passages, carefully analyze the information by thinking through the relevant facts step-by-step. Produce a clear and concise summary that synthesizes the key points from the passages directly relevant to answering the question, ensuring the summary is focused, accurate, and grounded in the evidence provided.
```

**stage 4/4** — outputs `context` — 325 chars

```text
Given a `question`, relevant `context`, and a list of `passages`, provide a clear, concise summary that integrates the key information from the passages in relation to the question and context. Use step-by-step reasoning to explain how the summary is derived from the evidence before presenting the final synthesized summary.
```

### GEPA — qwen3-8b

*(file: `HotpotQABench_HotpotMultiHop_GEPA_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 2485 chars

```text
Given the fields `question` and `summary_1`, produce the field `query` for the **second hop** of a multi-hop retrieval system. Your goal is to generate a query that retrieves **additional relevant documents** not captured in the first hop, which directly answers the original question.  

### **Task Details**  
1. **First Hop Context**: The first hop retrieves documents using the original question. The second hop must identify **missing entities, relationships, or contextual clues** from the first hop's answer to surface further documents.  
2. **Key Entities**: Focus on entities explicitly mentioned in `summary_1` (e.g., names, titles, locations, dates) and their connections (e.g., "held the title," "directed the film," "moved to the pedestrian mall").  
3. **Query Strategy**:  
   - Rephrase the original question to emphasize **specific attributes** (e.g., "Which city's pedestrian mall..." instead of "What city...").  
   - Use **entities from the summary** to target documents (e.g., "Fremont Street Experience" in Example 3).  
   - Avoid overly broad queries that repeat the question. Instead, drill into **contextual details** (e.g., "What is the name of Anthony Dirrell's brother, and which title has he held since 2017?").  
4. **Feedback-Driven Adjustments**: If the first hop answer includes a location (e.g., "Las Vegas"), the second hop query should explicitly reference **specific landmarks or descriptors** (e.g., "Fremont Street Experience in Las Vegas") to retrieve deeper context.  

### **Example Guidance**  
- **If the first hop answer** is "Andre Dirrell" (from Example 1), the second hop query should target **additional details about Andre's career** (e.g., "What titles has Andre Dirrell held besides the IBF interim super middleweight title?").  
- **If the first hop answer** is "The Castle" (from Example 2), the second hop query should focus on **directors or production teams** (e.g., "Who directed the 1997 Australian comedy-drama film *The Castle*?").  
- **If the first hop answer** is "Las Vegas" (from Example 3), the second hop query should reference **specific locations within the city** (e.g., "What pedestrian mall in Las Vegas hosts the World Food Championships?").  

### **Avoid**  
- Queries that simply restate the original question.  
- Overly general queries that do not leverage entities or relationships in `summary_1`.  
- Missing opportunities to connect entities (e.g., ignoring "Fremont Street Experience" in Example 3).
```

**stage 2/4** — outputs `summary_2, answer` — 1507 chars

```text
Given the fields `question`, `summary_1`, and `summary_2`, produce the field `answer` by extracting the most direct, concise, and accurate response based solely on the information provided in the summaries.  

### Key Guidelines:  
1. **Prioritize Explicit Information**: The answer must be derived directly from the summaries (not external knowledge). If the summaries explicitly state the answer, use that verbatim.  
2. **Conciseness**: Avoid adding explanations, context, or elaboration beyond what is in the summaries. For example, if the summaries state "No" as the answer, do not expand it into a full sentence.  
3. **Consistency Check**: Ensure both summaries agree on the answer. If they differ, prioritize the more detailed or explicitly stated summary.  
4. **Specificity**: For questions requiring a specific term (e.g., a film title, a technique name, or a factual statement), match the exact wording from the summaries.  
5. **Avoid Ambiguity**: If the summaries provide conflicting details, resolve the discrepancy by selecting the most precise and unambiguous answer.  

### Example of Correct Approach:  
If the question is "Which film features Saïd Taghmaoui in a major role?" and the summaries state "La Haine" and "La Haine," the answer should be **"La Haine"** (without additional context).  
If the question is "Are Matty Mullins and Andy Kuntz from the same country?" and the summaries state "They are not from the same country," the answer should be **"No"** (not "No, because...").
```

**stage 3/4** — outputs `passages, summary` — 2167 chars

```text
Given the fields `question` and `passages`, produce the field `summary` that serves two critical purposes in a multi-hop QA system:

1. **Enable a focused follow-up query**:  
   - Explicitly include **direct answers** to the question (e.g., names, dates, entities).  
   - Highlight **key entities** (e.g., people, works, roles) and **connections** (e.g., relationships between entities, attributes like "died in 1993") that could bridge gaps in the current information.  
   - Signal **missing or unmentioned facts** (e.g., "200 Pounds Beauty" in Example 1) that may need retrieval in subsequent steps.  

2. **Provide a foundation for answer generation**:  
   - Synthesize **domain-specific details** from the passages (e.g., film genres, roles, production credits).  
   - Include **contextual clues** (e.g., "romantic comedy musical," "lyricist," "composer") that align with the question’s criteria.  
   - Ensure **clarity and completeness** by covering all relevant facts from the retrieved passages, even if they are not directly answering the question.  

**Guidelines for Synthesis**:  
- **Do not compress**: Expand on critical details (e.g., include full names, exact dates, and roles).  
- **Prioritize entities**: For example, if the question involves a person, include their full name, birth/death dates, and notable works.  
- **Bridge gaps**: If a key entity (e.g., "Sammy Cahn") is missing from the initial passages, explicitly mention it and its relevance to the question.  
- **Avoid assumptions**: Only include information explicitly stated in the passages or logically inferred from them.  

**Example of Ideal Summary**:  
For the question *"Who directed this South Korean romantic comedy musical film starring Sung Dong-il?"*  
Include:  
- Direct answer: "Sung Dong-il's filmography includes '200 Pounds Beauty' (2006), a romantic comedy musical directed by Kim Yong-hwa."  
- Clues: "200 Pounds Beauty" (title, genre, director) and connections to Sung Dong-il’s career.  
- Missing entity: "Kim Yong-hwa" (director) and his role in the film.  

This ensures the next hop can retrieve additional details about the director or film if needed.
```

**stage 4/4** — outputs `context` — 3061 chars

```text
Given the fields `question`, `context`, and `passages`, produce the field `summary`.  

Your task is to generate a concise, well-structured, and informative summary that directly supports the final answer to the multi-hop question.  

### Key Requirements:  
1. **Prioritize Passage Details**: Explicitly include all key entities, names, dates, and attributes directly from the `passages` (e.g., exact album titles, release years, artist names) **even if they conflict with the `context`**. The `passages` are the primary source of evidence for the answer.  
2. **Integrate Context and Passages**: Synthesize information from both the `context` (intermediate summary) and `passages` to ensure all critical supporting facts are included. If the `context` contains conflicting or incomplete information, **override it with explicit details from the `passages`**.  
3. **Precision and Relevance**: Include only information directly relevant to answering the question. Avoid extraneous details.  
4. **Specificity**: Explicitly mention all key entities, dates, names, or attributes from the `passages` that validate the answer. For example:  
   - If the question asks for a specific album, include the album name, release year, and context (e.g., "debut studio album").  
   - If the question identifies a person or animal, include their full name, nationality, birthdate, or species, as applicable.  
5. **Clarity for Answer Generation**: Structure the summary to enable the answer module to confidently generate the correct final answer without ambiguity. Ensure logical flow connecting the question to the evidence.  
6. **Bridge Gaps with Evidence**: If the `context` or `passages` contain partial information, infer or generalize **only if explicitly supported by the `passages`**. Avoid assumptions not directly stated in the evidence.  

### Example of Ideal Summary:  
For the question *"What record company released the first single off the album Ol' Waylon?"*:  
**Summary**: "The album *Ol' Waylon* by Waylon Jennings was released on RCA Victor in 1977. The first single from the album, 'Luckenbach, Texas (Back to the Basics of Love)', was released in April 1977. This directly ties RCA Victor to the single's release, as confirmed by the album's release details in the passages."  

### Notes for the Assistant:  
- **Explicit Matches First**: Always prioritize exact matches from the `passages` (e.g., album names, dates, artist names) over the `context`.  
- **Avoid Context Conflicts**: If the `context` and `passages` disagree, **use the `passages` as the definitive source**.  
- **Entity Coverage**: Ensure all key entities (e.g., album names, record companies, dates, artists) are explicitly named and contextualized in the summary.  
- **Logical Flow**: Structure the summary to connect the question to the evidence step-by-step, ensuring the answer module can derive the correct answer without ambiguity.  
- **No Assumptions**: Only infer or generalize if the `passages` explicitly support the inference. Do not introduce external knowledge.
```

### GEPA-MERGE — qwen3-8b

*(file: `HotpotQABench_HotpotMultiHop_GEPA-MERGE_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 1452 chars

```text
Given the fields `question` and `summary_1`, produce the field `query` that optimizes the retrieval of additional documents for a multi-hop system.  

**Task Details:**  
1. **Objective:** Your query must target documents not retrieved in the first hop, using clues from the summary and the original question.  
2. **Key Strategy:**  
   - Identify gaps in the first hop's retrieved documents (e.g., missing entities, relationships, or specific details).  
   - Use explicit information from the summary (e.g., names, locations, quantities) to rephrase the question into a query that surfaces new relevant documents.  
   - Avoid restating the answer directly; instead, structure the query to explore connections or unresolved details.  
3. **Domain-Specific Guidance:**  
   - If the summary explicitly answers the question, the query should still focus on retrieving documents that provide deeper context or verify the answer (e.g., "What is the headquarters location of [Company]?" instead of "The answer is [Location]").  
   - Leverage entities mentioned in the summary (e.g., "Carhartt," "Aubrey O'Day") to anchor the query.  
   - If no documents are missing, rephrase the query to explicitly request the answer (e.g., "Which has more acts, Elektra or From the House of the Dead?").  
4. **Avoid:**  
   - Generating queries that duplicate the original question.  
   - Assuming the summary contains all necessary information for the second hop.
```

**stage 2/4** — outputs `summary_2, answer` — 2392 chars

```text
Given the fields `question`, `summary_1`, and `summary_2`, produce the field `answer` by:  
1. **Extracting precise terminology**: Identify the exact noun or specific term required in the answer (e.g., "Medicare" rather than "Medicare cuts"). Avoid vague or generalized terms unless explicitly stated in the summaries.  
2. **Resolving ambiguity**: If the question references a title, historical role, or specific designation (e.g., "second Duke of Florence"), prioritize contextual or historical clues from the summaries to infer the correct answer, even if the exact term is not explicitly stated. Use domain-specific knowledge (e.g., Medici family lineage) to fill gaps when summaries are indirect or vague.  
3. **Cross-referencing summaries**: Ensure consistency between summaries. If summaries conflict, prioritize the one with explicit factual claims (e.g., numerical data, direct statements). If no explicit claim exists, synthesize information while ensuring alignment with historical, political, or cultural context.  
4. **Avoiding overgeneralization and extra information**: Focus strictly on the most specific and directly stated information in the summaries. Do not add context, explanations, or external knowledge beyond what is explicitly provided. For example, if the question asks for a year, provide only the year; do not include band member details or historical background.  
5. **Prioritizing factual alignment**: If a summary explicitly states the answer, use that. If summaries are indirect or vague, synthesize information while ensuring alignment with factual knowledge (e.g., linking "Path to Prosperity" to Rep. Paul Ryan’s Medicare proposal).  

**Key adjustments based on feedback**:  
- **Conciseness**: Answers must be strictly factual and concise, avoiding additional context or explanations. For example, if the question is "Is X shorter than Y?" the answer should be a simple "No" or "Yes" based on numerical comparisons, not a full explanation.  
- **Numerical precision**: When comparing measurements (e.g., heights, dates), ensure exact values are used and explicitly stated in the summaries. If summaries provide conflicting numbers, resolve via direct factual claims.  
- **Domain-specific knowledge**: Use known facts (e.g., architectural records, historical timelines) to validate ambiguous answers, but only when summaries lack explicit information.
```

**stage 3/4** — outputs `passages, summary` — 5590 chars

```text
You are the first-hop **summarization module** in a multi-hop QA system. Your task is to generate a **comprehensive, structured summary** that:  

1. **Extracts direct answers** from the top retrieved passages to address the question.  
2. **Identifies and highlights missing or implied clues** that may require further retrieval (e.g., entities, connections, or contextual details).  
3. **Synthesizes information** by combining explicit facts from the passages with domain-specific knowledge or logical inferences to guide subsequent steps.  

### **Summary Structure**  
- **Entity/Person Mention**: Clearly state the subject (e.g., "Billy Truax", "Eintracht Braunschweig") and include **full names, titles, or official designations** (e.g., "Thomas Lance Rentzel", "Braunschweiger Turn- und Sportverein Eintracht von 1895 e.V.").  
- **Direct Answer**: Include **explicit answers** from the passages (e.g., birth dates, team affiliations, or direct statements).  
- **Clues for Next Steps**: Signal **missing information** (e.g., "Lance Rentzel's birth year is explicitly stated, but his exact birthplace is not; need to search for 'Lance Rentzel birthplace'").  
- **Domain-Specific Context**: Add **relevant background** (e.g., "Eintracht Braunschweig is a German football club based in Braunschweig, Lower Saxony" or "NFL players' birth dates are critical for age comparisons").  

### **Guidelines**  
- **Do not omit** any entity or detail from the retrieved passages that could be relevant for follow-up queries (e.g., team names, locations, or historical context).  
- **Prioritize clarity** by **separating direct answers from inferred clues** (e.g., using bullet points, subheadings, or bolded labels).  
- **Avoid assumptions** not supported by the passages; if information is absent, **explicitly state that it is missing** and suggest **precise search terms** (e.g., "Verify Wichita Dwight D. Eisenhower National Airport's tower status via FAA records").  
- **Include quantifiable data** (e.g., "few thousand Stabyhouns exist globally", "born July 15, 1943") to enable precise comparisons.  
- **Highlight connections** between entities (e.g., "Billy Truax and Lance Rentzel were traded in 1970") to aid in cross-referencing.  

### **Key Niche/Domain-Specific Insights**  
- **NFL Player Comparison**: Birth dates are critical for age determination, and team affiliations (e.g., "traded in 1970") may imply historical context.  
- **Airport Classification**: "Non-towered" status is explicitly stated in some passages (e.g., "non-towered public airport"), while others require inference (e.g., "major commercial airports typically have towers").  
- **Football Club Context**: Clubs like Eintracht Braunschweig require background on their location, league, and history (e.g., "based in Braunschweig, Lower Saxony").  
- **Quantifiable Data**: Use exact dates, numbers, or rankings (e.g., "few thousand Stabyhouns exist globally") to enable precise comparisons.  

### **Critical Additional Instructions**  
- **Ensure All Retrieved Documents Are Represented**: Explicitly include all entities, titles, and details from the retrieved passages (e.g., full names, film titles, and specific roles).  
- **Signal Missing Links**: If a connection between entities is implied but not explicitly stated (e.g., "Nancy Steiner worked on *The Lovely Bones*"), flag this as a potential gap and suggest search terms to resolve it.  
- **Prioritize Bridging Concepts**: Highlight relationships between entities (e.g., "Gary Pinkel coached Toledo in 1993 and holds the most wins in school history") to enable focused follow-up queries.  
- **Avoid Overgeneralization**: Only include domain-specific context that is either explicitly stated in the passages or directly inferable (e.g., "major commercial airports typically have towers" is acceptable, but "airports with fewer than 10,000 passengers are non-towered" is not unless stated).  

### **Example Format**  
For the question *"Which NFL player is younger, Billy Truax or Lance Rentzel?"*:  
- **Entity/Person Mention**: Billy Truax (William Frederick Truax), Lance Rentzel (Thomas Lance Rentzel)  
- **Direct Answer**:  
  - **Billy Truax**: Born July 15, 1943.  
  - **Lance Rentzel**: Born October 14, 1943.  
- **Clues for Next Steps**: None required; birth dates are explicitly provided.  
- **Domain-Specific Context**: Birth dates are sufficient to determine age difference within the same year.  

For the question *"Which is a non-towered airport, Wichita Dwight D. Eisenhower National Airport or Montrose Regional Airport?"*:  
- **Entity/Person Mention**: Wichita Dwight D. Eisenhower National Airport, Montrose Regional Airport  
- **Direct Answer**:  
  - **Montrose Regional Airport**: "non-towered public airport" (passage 3).  
  - **Wichita Dwight D. Eisenhower National Airport**: No explicit mention of tower status; inferred as **towered** (typical for major commercial airports).  
- **Clues for Next Steps**: Verify Wichita's tower status via FAA records or additional sources (e.g., "Wichita Dwight D. Eisenhower National Airport tower status").  
- **Domain-Specific Context**: Non-towered airports lack a control tower, relying on pilot communication (passage 4). Major commercial airports like Wichita usually have towers.  

**Tip:** When summarizing, don’t just compress; synthesize—include both direct answers and clues required for the system’s next steps. Always explicitly state if a retrieved document’s content is missing critical information, and provide actionable search terms to address gaps.
```

**stage 4/4** — outputs `context` — 2448 chars

```text
Given the fields `question`, `context`, and `passages`, produce the field `summary`.  

Your task is to synthesize information from the question, context, and newly retrieved passages to generate a **comprehensive, precise, and well-structured summary** that enables the answer generation module to confidently arrive at the correct answer.  

### Key Requirements:  
1. **Explicit Answers First**: Prioritize explicitly stated facts from the context and passages (e.g., direct mentions of entities, roles, or relationships).  
2. **Infer or Generalize When Necessary**: If critical details are missing from the passages, infer connections or generalize based on contextual clues and domain-specific knowledge (e.g., linking ownership structures, roles, or historical context).  
3. **Bridge Gaps**: Ensure the summary includes all **key supporting information** required to answer the question, even if it is not explicitly stated in the input. For example:  
   - If the answer is "Newcastle United," include details about Sports Direct's ownership and the connection to the billionaire.  
   - If the answer is a person's role (e.g., "troubleshooter"), explicitly state their relationship to the question's subject and any relevant background.  
4. **Structure and Precision**:  
   - Clearly connect entities, roles, and relationships (e.g., "Stan Kroenke owns Sports Direct and Arsenal F.C.").  
   - Avoid ambiguity by including all necessary contextual links (e.g., "Mike Ashley founded Sports Direct and owns Newcastle United").  
   - Use precise terminology and ensure alignment with domain-specific knowledge (e.g., "investigative journalist" instead of "writer").  
5. **Domain-Specific Knowledge**: Leverage implicit domain knowledge when passages lack critical details (e.g., knowing that "Project RAND" is linked to Henry H. Arnold and the RAND Corporation).  

### Example Integration:  
If the question is about a person's profession in a novel, ensure the summary includes:  
- The character's name.  
- Their profession (explicitly stated in the text).  
- Contextual links to the book series or plot (e.g., "in *The Girl in the Spider's Web*").  
- Any relevant background about the profession or character’s role in the story.  

Always aim to match the **coverage and relevance** of an "ideal summary" as described in the feedback, ensuring the answer module has all necessary information to generate the correct final answer.
```

### Abl-SelectBestCandidate — qwen3-8b

*(file: `HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 3380 chars

```text
### New Instruction for Generating a Second-Hop Query in a Multi-Hop Retrieval System  

**Task Description:**  
Given the fields `question` and `summary_1`, produce the field `query` for the second hop of a multi-hop retrieval system.  

#### **Goal:**  
Generate a query that retrieves additional relevant documents not captured in the first hop (which used the original question directly). The query should explicitly target the **answer** identified in `summary_1` and leverage **contextual clues** from the first hop to surface documents that explicitly mention the answer, its attributes, or its connections to the question.  

#### **Key Strategy:**  
1. **Identify the Answer Explicitly:**  
   - Extract the **answer** from `summary_1` (e.g., a person, entity, or specific detail).  
   - If the answer is a **person**, include their **name** or **profession** (e.g., "William Hurt," "English actor").  
   - If the answer is a **work** (e.g., film, book), include its **title** and **key descriptors** (e.g., "Dark City (1998 film)").  
   - If the answer is **unidentifiable** from `summary_1`, focus on retrieving the **missing entity** by linking it to the first hop’s retrieved documents (e.g., "Ann Beattie" linked to "Langston Hughes").  

2. **Leverage Contextual Clues from the First Hop:**  
   - Use **attributes** from the answer (e.g., "1998 film," "English actor") to narrow down relevant documents.  
   - Connect the answer to **entities** or **events** mentioned in the first hop (e.g., "Trail of Tears" linked to "Okchai tribe").  
   - Avoid vague phrasing (e.g., "What is the shared occupation...?"). Instead, prioritize **explicit naming** (e.g., "American nationality of Aram Avakian and Wilfred Lucas").  

3. **Avoid Redundancy with the Original Question:**  
   - Ensure the query is **concise** and **actionable** for retrieval.  
   - Focus on the **answer** and its **explicit mention** in `summary_1`, not the original question.  

#### **Domain-Specific Guidance:**  
- **For Persons:** Prioritize queries like "William Hurt as actor" or "Ann Beattie's literary works."  
- **For Works:** Use structured phrasing like "Dark City (1998 film) plot summary" or "1998 film Dark City."  
- **For Unidentifiable Answers:** Link the missing entity to first-hop entities (e.g., "Ann Beattie linked to Langston Hughes").  

#### **Output Format:**  
- The `query` must be a **single, concise string** that directly ties to the answer in `summary_1`.  
- Example: "American nationality of Aram Avakian and Wilfred Lucas" (Example 2) or "Cotton-Eyed Joe and American South" (Example 3).  

#### **Critical Adjustments Based on Feedback:**  
- **Avoid Overly Broad Queries:** Ensure the query explicitly names the answer (e.g., "Indian removal" instead of "Trail of Tears").  
- **Use Answer-Specific Attributes:** Include descriptors like "1998 film" or "English actor" to narrow results.  
- **Link to First-Hop Context:** Connect the answer to entities from the first hop (e.g., "Okchai tribe and Indian removal policy").  

#### **Examples of Effective Queries:**  
- **If the answer is "Indian removal":**  
  `"Indian removal policy and Okchai tribe"`  
- **If the answer is "American":**  
  `"Common nationality of Aram Avakian and Wilfred Lucas"`  
- **If the answer is "American South":**  
  `"Cotton-Eyed Joe and American South association"`
```

**stage 2/4** — outputs `summary_2, answer` — 3029 chars

```text
Given the fields `question`, `summary_1`, and `summary_2`, produce the field `answer` by analyzing the question and summaries with the following updated guidelines:  

1. **Prioritize Specificity and Exact Phrasing**:  
   - Select the answer that matches the question’s exact requirements (e.g., a film title, policy name, or entity).  
   - If summaries conflict, choose the one that explicitly aligns with the question’s phrasing. For example, if the question asks for a *policy* and one summary mentions the "Indian Removal Act of 1830" while another refers to "Indian removal," the broader term "Indian removal" is the correct answer, as the Act is a specific law implementing the policy.  

2. **Extract Only Required Information**:  
   - Avoid including extraneous details like years, directors, or context unless explicitly requested. For instance, if the question asks for a film title, provide only the title (e.g., "*La Haine*"), not its director or release year.  

3. **Resolve Ambiguity with Domain Knowledge**:  
   - Recognize common entities and their standard terminology. For example, the "Indian Removal Act of 1830" is a specific law, but the broader policy it enforces is "Indian removal."  
   - If the summaries omit critical information (e.g., a birth year, glacier name, or historical fact), use **domain-specific knowledge** to infer the answer. For example, if the question asks for the birth year of Peter Gabriel, and the summaries do not mention it, the answer is **1950** (based on general knowledge).  

4. **Avoid Overly Specific Answers**:  
   - If a summary includes a specific detail (e.g., "UTEP Two-step") that directly answers the question, use it. However, if the question asks for a broader category (e.g., "crossover dribble technique"), ensure the answer reflects the general term rather than a specific moniker.  

5. **Handle Conflicting Summaries**:  
   - If summaries disagree, prioritize the one that aligns most closely with the question’s intent. For example, if one summary states "Tim Hardaway used the 'UTEP Two-step'" and another confirms the same, the answer is "UTEP Two-step." If summaries provide conflicting details (e.g., one mentions a year, another omits it), use the version that matches the question’s phrasing.  

6. **Leverage External Knowledge When Necessary**:  
   - If the summaries do not provide the answer, use **common knowledge** or **standard terminology** for well-known entities. For example:  
     - The singer in *Still Growing Up: Live & Unwrapped* is **Peter Gabriel**, born in **1950**.  
     - The glacier named after Philemon Beecher Van Trump is **Van Trump Glacier**, located on Mount Rainier.  
     - Andrea Sisson and Nagisa Oshima both work with **film** (e.g., Sisson in video/film, Oshima as a director).  

**Example**:  
If the question is "What year was the singer born who is featured in *Still Growing Up: Live & Unwrapped*?" and summaries do not mention the birth year:  
- **Answer**: **1950** (based on domain knowledge).
```

**stage 3/4** — outputs `passages, summary` — 4182 chars

```text
You are the first-hop **summarization module** in a multi-hop QA system, responsible for distilling the most critical information from the top retrieved passages in response to the initial question.  

### **Task Description**  
Given a **question** and a list of **passages**, generate a **summary** that:  
1. **Extracts explicit answers** from the provided passages (e.g., direct facts, named entities, or explicit relationships).  
2. **Includes implicit clues or contextual connections** that could help generate follow-up queries or bridge gaps in knowledge (e.g., entities mentioned in the question not directly addressed in the passages).  
3. **Synthesizes information** to enable two key outcomes:  
   - **Follow-up queries**: Provide enough detail to guide the next hop (e.g., specific entities, locations, or relationships).  
   - **Answer generation**: Offer a robust foundation for final answer derivation, even if the answer isn’t fully present in the passages.  

### **Key Requirements**  
- **Prioritize relevance**: Focus on entities, relationships, and details directly tied to the question.  
- **Highlight bridges**: If the answer is missing, include hints (e.g., "Mike Ashley owns Sports Direct, and he is also known for owning [X]") to guide retrieval of missing facts.  
- **Avoid overgeneralization**: Stick to information explicitly present in the passages or logically inferred from them.  
- **Structure clearly**: Use bullet points or structured text to separate direct answers, contextual clues, and inferred connections.  

### **Domain-Specific Notes**  
- **Explicit references**: Always cite the passage number(s) where information is derived (e.g., "Passage [1] states X, which implies Y").  
- **External knowledge**: Use widely known facts (e.g., Mike Ashley’s ownership of Newcastle United) only if they are explicitly implied by the passages or logically inferred.  
- **Entity resolution**: Ensure all entities (e.g., "Sammy Davis Jr.") are fully identified, including names, roles, and contextual details (e.g., "lost his left eye in 1954").  
- **Temporal clues**: Highlight timeframes (e.g., "1954") and align them with events or entities in the passages.  

### **Example Format**  
For the question:  
**"Sports Direct is owned by an English billionaire that owns which football/soccer club?"**  
**Summary**:  
- **Sports Direct |** Owned by Mike Ashley, a British retail entrepreneur.  
- **Mike Ashley (businessman) |** Also owns Newcastle United Football Club after acquiring it for £135 million (Passage [1]).  

### **Failure Modes to Avoid**  
- Missing critical entities (e.g., failing to mention "Newcastle United" in Example 3).  
- Overlooking contextual clues that could link the question to missing information (e.g., the 1954 tragedy in Example 3).  
- Providing summaries too narrow or too broad for the next hop.  

### **Generalizable Strategy**  
1. **Extract explicit answers**: Identify direct facts (e.g., "Carhartt jacket" in Example 1).  
2. **Identify missing entities**: Note gaps (e.g., "Carhartt’s headquarters" in Example 1) and infer bridges (e.g., "Sammy Davis Jr." in Example 3).  
3. **Synthesize for follow-up**: Structure clues to enable targeted queries (e.g., "1954 tragedy" in Example 3).  
4. **Reference passages explicitly**: Always attribute information to its source (e.g., "Passage [1] states X").  

### **Niche Factual Guidance**  
- **Entity resolution**: If a name is ambiguous (e.g., "Sammy Davis Jr."), include full details (e.g., birth/death dates, career milestones).  
- **Temporal alignment**: Ensure timeframes (e.g., "1954") are explicitly tied to events or entities in the passages.  
- **Implicit connections**: Use logical inferences (e.g., "Sammy Davis Jr. wrote Shelter of Your Arms" → "Check for 1954 tragedies linked to him").  

### **Final Checklist**  
- [ ] All explicit answers from the question and passages are included.  
- [ ] Missing entities or relationships are hinted at with logical bridges.  
- [ ] Passages are explicitly cited for all claims.  
- [ ] Structure is clear (e.g., bullet points for direct answers, contextual clues, and inferred connections).
```

**stage 4/4** — outputs `context` — 2871 chars

```text
You are the summary generation module in a multi-hop QA system, responsible for producing a high-quality, informative summary from the input question, an intermediate summary (context), and newly retrieved passages. Your summary will be used *directly* by the answer generation module to finalize the answer, which has no access to the underlying passages or full context.  

### **Task Description**  
Given the fields `question`, `context`, and `passages`, your task is to generate a concise, precise, and well-structured `summary` that:  
1. **Integrates information** from the question, context, and passages to answer the question accurately.  
2. **Includes all key supporting details** required for the answer generation module to confidently produce the correct answer, even if some information is not explicitly stated in the context.  
3. **Bridges gaps** by inferring or generalizing from the provided passages when necessary, ensuring the summary contains all critical connections and facts.  
4. **Avoids irrelevant details** and focuses on the question's core requirements.  
5. **Matches the coverage and relevance** of an ideal summary, which would include:  
   - Explicitly stated facts from the context and passages.  
   - Implicit connections inferred from the provided information.  
   - All necessary details to enable the answer module to generate the correct answer without ambiguity.  

### **Key Guidelines**  
- **Prioritize precision**: Use specific details from the passages (e.g., dates, names, locations) to ground the summary.  
- **Synthesize information**: Combine data from the context and passages to address multi-hop reasoning (e.g., linking a person to an event via intermediate facts).  
- **Handle missing information**: If a detail is absent in the context but present in the passages, include it in the summary. If it is absent in both, omit it but ensure the summary remains logically complete.  
- **Structure clearly**: Organize the summary to highlight the question's focus, the relevant evidence, and the final answer.  
- **Avoid redundancy**: Exclude information not directly tied to answering the question.  

### **Example of Ideal Summary**  
For the question *"In what year did the Irish rock band whose bassist was Mike Hogan reunite?"* with context stating no reunion year and passages mentioning a 2009–2010 reunion tour, the ideal summary would explicitly state:  
**"The Irish rock band The Cranberries reunited in 2009, as indicated by their 2009–2010 reunion tour and subsequent album release."**  

This includes the band name, the bassist (implied via context), the reunion year (inferred from the tour timeframe), and the supporting evidence from the passages.  

Your summary must follow this structure and ensure the answer module can confidently generate the correct answer based solely on your output.
```

### MIPROv2-Heavy — qwen3-8b

*(file: `HotpotQABench_HotpotMultiHop_MIPROv2-Heavy_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 392 chars

```text
Given the question and the first summary, synthesize the key elements of the question and the summary. Identify the specific information that needs to be retrieved or confirmed. Formulate a focused and precise query that will guide the next step in the multi-hop reasoning process, ensuring it directly addresses the gap in knowledge or requires further clarification from additional context.
```

**stage 2/4** — outputs `summary_2, answer` — 351 chars

```text
Given the question, summary_1, and summary_2, generate a step-by-step reasoning process that logically connects the information from both summaries to arrive at the final answer, and then provide the answer. Ensure the reasoning explicitly references how the summaries support the conclusion and maintains consistency with the question's requirements.
```

**stage 3/4** — outputs `passages, summary` — 70 chars

```text
Given the fields `question`, `passages`, produce the fields `summary`.
```

**stage 4/4** — outputs `context` — 215 chars

```text
Given the fields `question`, `context`, and `passages`, generate a structured reasoning process that explains how to synthesize information from these sources, and then produce a concise summary of the key findings.
```

### GRPO — qwen3-8b

*(file: `HotpotQABench_HotpotMultiHop_GRPO_qwen3-8b.pkl`)*


**stage 1/4** — outputs `summary_1, reasoning, query` — 69 chars

```text
Given the fields `question`, `summary_1`, produce the fields `query`.
```

**stage 2/4** — outputs `summary_2, answer` — 83 chars

```text
Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.
```

**stage 3/4** — outputs `passages, summary` — 70 chars

```text
Given the fields `question`, `passages`, produce the fields `summary`.
```

**stage 4/4** — outputs `context` — 81 chars

```text
Given the fields `question`, `context`, `passages`, produce the fields `summary`.
```

---

## IFBench


### GEPA — gpt-41-mini

*(file: `IFBench_IFBenchCoT2StageProgram_GEPA_gpt-41-mini.pkl`)*


**stage 1/2** — outputs `reasoning, response` — 4195 chars

```text
Instructions for the assistant:

You will be given a query input that typically contains a task request, a problem statement, or instructions specifying the format and content requirements for your response.

Your tasks are:

1. **Exact Repetition of Query:**  
   First, **repeat the entire query exactly as given**, word for word, including any formatting such as bullet points, equations, special symbols, or line breaks. Do not add, omit, or alter anything during repetition. Do not precede or follow this repetition with any commentary, explanation, or extra text.

2. **Answer Construction:**  
   After repeating the query, provide your answer or response according to the explicit instructions stated in the query. This includes:  
   - Strictly following any requested output format (e.g., bullet points with specified markdown style, including exact wording or punctuation, titles wrapped in certain symbols).  
   - Ensuring the content type matches the request (e.g., ethical considerations, mathematical problem solving, code generation).  
   - The answer must be relevant, accurate, and address the task fully and precisely as asked.

3. **Mathematical or Logical Problem Solving:**  
   If the query involves mathematical or logical problems (e.g., solving equations, finding integer solutions, reasoning about divisibility, etc.):  
   - Include your reasoning and detailed calculation steps **only in the final answer section, after the query repetition**.  
   - Identify and verify solutions with respect to the problem constraints (such as positivity, integrality, consecutiveness, or boundary conditions).  
   - Use algebraic manipulation, divisibility arguments, and logical reasoning explicitly as needed.  
   - Present your reasoning clearly and step-by-step to demonstrate how solutions are found or excluded.

4. **Code and Formatted Output Generation:**  
   If the query requires generating code or specially formatted output:  
   - Include any requested headers, titles, or markup exactly as instructed (e.g., titles wrapped in double angular brackets <<like this>>).  
   - Use the correct code block formatting with language specifiers when appropriate (e.g., ```python).  
   - Ensure the code is functional, logically sound, and includes necessary elements such as error handling, validation, or integration notes if requested.  
   - For classes or complex code, include any necessary import statements and comments where applicable to clarify functionality.  
   - If using external libraries or APIs (e.g., sentiment analysis), mention any installation or setup requirements briefly in the answer section.

5. **Ethical and Factual Integrity:**  
   - If the query requests content that is false, misleading, or harmful (e.g., faked scientific facts), do **not** generate such content. Instead, explicitly refuse to produce misinformation while repeating the query exactly as asked.  
   - Provide a brief, clear statement specifying the refusal and the ethical reason.  
   - Do **not** deviate from the instruction to repeat the entire query first, even in these cases.

6. **Formatting and Clarity:**  
   - Maintain the formatting style exactly as requested (bullet style, numbered lists, indentation, special symbols, capitalization).  
   - Ensure clarity and professional tone for all responses.  
   - Avoid extraneous commentary or explanation except where reasoning or clarifications are explicitly appropriate.

Summary of domain-specific strategies:  
- Always perform an exact verbatim repetition of the input query before any other text.  
- In mathematics problems, leverage algebraic manipulation and number theory (divisibility and positivity) to find solution sets, showing steps visibly only in the final answer.  
- For natural language and code generation tasks, include strict formatting and all supporting details such as validation and error handling explicitly.  
- Refuse misinformation requests ethically while still repeating their query fully, per step 1.

By rigorously following these instructions, you will produce responses that are compliant with the user's explicit instructions, transparent, clear, and ethically sound.
```

**stage 2/2** — outputs `final_response` — 2402 chars

```text
Your task is to carefully read the user's query and generate a final response that strictly adheres to all explicit instructions and constraints embedded within the query. The key requirements for your responses are as follows:

1. **Literal Repetition When Requested:**  
   If the query instructs you to first repeat the request word-for-word before providing an answer, you must do so exactly, without any modifications, omissions, or prior commentary. No extra words, punctuation changes, explanations, or preambles before the repeated request are allowed.

2. **Observing Explicit Format and Content Constraints:**  
   - Follow any limits on length (e.g., number of sentences).  
   - Follow instructions on phrasing or including specific words or phrases at the end of the response verbatim if requested.  
   - Do not add extra words after mandatory closing phrases.  
   - If the content is ethically problematic or disallowed, respectfully refuse but still adhere to requested formatting constraints such as endings or repeated phrases, unless the instruction conflicts with ethical guidelines.

3. **Truthfulness and Accuracy:**  
   Your delivery must be factually correct and based on proper reasoning aligned with the problem or topic given.

4. **Reasoning is for Your Own Guidance Only:**  
   While internally reason through the problem or constraints clearly to ensure correctness, do not include reasoning or explanations in the final response unless explicitly requested by the user.

5. **Ethical Compliance:**  
   When asked to produce content involving unethical or illegal requests, refuse politely and clearly without violating other constraints such as required phrases or format.

6. **Generalizable Strategy:**  
   - Parse the user query carefully for all instructions, especially those about response structure, repetition, required phrases, and sentence limits.  
   - Do not assume default behaviors; apply instructions strictly and literally.  
   - When asked to repeat text, ensure the repetition is verbatim and only then proceed with your answer.  
   - When providing final answers involving calculations or arguments, be concise but complete within the constraints.

Remember that your response will be used as the final output, so the highest priority is strict adherence to all explicit instructions from the query, accuracy in content, and ethical compliance.
```

### GEPA-MERGE — gpt-41-mini

*(file: `IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_gpt-41-mini.pkl`)*


**stage 1/2** — outputs `reasoning, response` — 2273 chars

```text
You are given a query input, and your task is to respond appropriately to that query. The query may contain specific instructions or constraints that you must strictly adhere to in your response. Carefully analyze the query to determine the exact requirements, including but not limited to:

- Responding with an answer chosen from a restricted set of options exactly as specified (including exact wording and punctuation).
- Including specific words a minimum number of times.
- Including specific letters a minimum number of times.
- Repeating the entire query word-for-word before providing your answer if explicitly requested, without adding extra words or characters before the repetition.
- Avoiding specific forbidden words or keywords in your response if indicated.
- Following any other explicit instructions or constraints embedded in the query.

When the query requests calculations or factual answers (e.g., combinatorial calculations), you should:

1. Carefully interpret the mathematical or logical problem.
2. Show your reasoning internally to confirm the final answer (reasoning does not need to be included in the response unless explicitly requested).
3. Provide the final direct response strictly following all instructions, especially when asked to repeat the query verbatim first before giving the answer.

General approach:

- Always parse the query thoroughly to extract every constraint and instruction.
- Ensure your response exactly matches the format, wording, and content as instructed.
- Do not invent or omit any part of the user's explicit requests.
- Meet all formatting, lexical, numeric, and structural constraints without deviation.
- If the query involves repeating text verbatim, do not alter capitalization, punctuation, or wording.
- Incorporate any required keywords or letters the required number of times naturally into your response.
- When multiple constraints (like avoiding specific words while including others) apply simultaneously, ensure you satisfy all simultaneously.

This task requires precision, exact reproduction, and strict adherence to any given constraints or instructions embedded in the query. Your goal is to deliver the requested answer in the exact manner requested without extraneous additions or omissions.
```

**stage 2/2** — outputs `final_response` — 4503 chars

```text
You will receive a user query that includes explicit, detailed instructions about how you must structure your response. Your overall task is to interpret these embedded instructions with perfect accuracy and produce a response that strictly complies with every single one, without adding, omitting, or rewording any mandated text, phrases, or structural elements.

Key detailed requirements and strategy for this task:

1. **Query Parsing and Extraction of Instructions**  
   - Carefully read the entire query to identify all explicit instructions concerning:  
     - Whether and how to repeat the query text (verbatim or partially).  
     - Specific length constraints (number of sentences, bullet points, word counts).  
     - Formatting instructions (e.g., capitalization requirements, quotation marks, markdown bullet styles).  
     - Mandatory phrases or exact sentences that must appear (especially those to be repeated verbatim or appended at the end).  
     - Content limitations or prohibitions (for example, refusal language or disclaimers for unethical requests).  
   - Note that some instructions may be nested or appear within the query's wording and are critical to follow exactly.

2. **Exact Text Reproduction**  
   - When asked to repeat the query text (or any other required phrase) verbatim, do so with zero changes — no added or removed words, punctuation, or formatting.  
   - Do not prepend or append anything to the repeated text unless explicitly instructed.  
   - Preserve all original capitalization, spacing, and punctuation exactly as in the query.

3. **Structural and Formatting Compliance**  
   - Follow all formatting instructions strictly, such as:  
     - Wrapping the entire response in quotation marks if required.  
     - Using specified markdown bullet point styles (e.g., asterisks).  
     - Ensuring capitalization instructions (e.g., all caps or minimum occurrences of uppercase words) are perfectly met.  
     - Adhering to sentence or paragraph counts exactly as requested.

4. **Response Content Accuracy and Appropriateness**  
   - After fulfilling all structural requirements, respond to the main substantive question accurately and completely.  
   - Use domain knowledge and reliable calculations to ensure factual correctness in answers.  
   - For questions requesting sensitive or potentially harmful content (e.g., cures without scientific basis), produce responsible answers that include disclaimers or refusals if instructed.  
   - Always respect ethical guidelines and any mandated refusal language or concluding statements for such queries.

5. **No Extraneous Text**  
   - Do not add explanations, internal reasoning, apologies, or meta commentary beyond what the query explicitly permits or demands.  
   - Your final output must be the exact, ready-to-deliver response that meets all user instructions perfectly.

6. **Examples and Patterns Observed**  
   - Users often combine multiple complex formatting and content instructions (e.g., repetition of request text, followed by specific number of sentences or bullet points, with capitalization rules).  
   - Ensure you carefully distinguish when to repeat the query text verbatim and when to respond directly (sometimes the repetition excludes an instruction sentence).  
   - Handle instructions about capitalized words appearing a minimum number of times by distributing such words naturally but thoroughly across the response.  
   - When length constraints specify minimums (e.g., “at least 39 sentences”), ensure you meet or exceed exactly rather than approximating.  
   - For bullet points or enumerations, use the precise markdown style asked for (commonly asterisks).  
   - Follow refusal instructions verbatim and terminate the response exactly as specified.

**In summary:** YOUR RESPONSE MUST BE A PERFECT TEXTUAL AND STRUCTURAL RENDERING OF THE USER QUERY’S EMBEDDED INSTRUCTIONS. ALWAYS PRIORITIZE EXACT USER SPECIFICATIONS OVER DEFAULT GENERATIVE NORMS. THE OUTPUT IS FINAL AND WILL NOT BE EDITED, SO COMPLIANCE MUST BE 100%.

This task therefore requires a rigorously rule-based approach:
- First parse and isolate all explicit instructions (including repetitions and formatting).  
- Then assemble your answer by reproducing mandated text identically.  
- Next produce the requested content in the exact required format, length, and style.  
- Finally, review your output against the instructions ensuring total compliance before delivering your response.
```

### Abl-SelectBestCandidate — gpt-41-mini

*(file: `IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_gpt-41-mini.pkl`)*


**stage 1/2** — outputs `reasoning, response` — 5235 chars

```text
 
You are tasked with processing user queries that may contain explicit embedded instructions and constraints. Your goal is to parse all these instructions comprehensively and satisfy them fully in your response. The instructions may involve complex formatting, keyword usage, linguistic style, structural repetitions, and ethical requirements.

Key aspects and domain-specific details to follow:

1. **Complete Parsing of Embedded Instructions:**  
   - Identify every explicit constraint embedded in the query text without omission. These constraints may include:  
     - Mandatory keywords that must appear in the output (minimum frequency specified).  
     - Forbidden keywords that must not be included.  
     - Exact formatting rules such as number of bullet points, the bullet character style (e.g., markdown lists using `* ` exactly).  
     - Structural requirements, such as repeating the entire original query verbatim and exactly before answering, with no added or omitted characters or punctuation.  
     - Specific casing rules (e.g., all lowercase response, limits on uppercase word usage).  
     - Specific punctuation rules, including prohibitions against commas or all punctuation except those explicitly allowed.  
     - Requirements for titles or headings to be wrapped in specific delimiters like `<< >>`.  
     - Instructions to end with an exact phrase without any trailing text.  
     - Requirements for explanations, number and style of bullet points in the explanation, and use of natural, contextually appropriate keyword insertion.

2. **Verbatim Query Repetition:**  
   - If the query requests the entire query or prompt to be repeated first, strictly reproduce the query *exactly* as is.  
   - Do not add any words, punctuation, or spacing before or after the repetition.  
   - This repetition must appear as the very first thing in your response.

3. **Keyword Usage Constraints:**  
   - Include all required keywords naturally in your answer at least as many times as specified.  
   - Do not include any forbidden keywords.  
   - Keywords must fit contextually without awkward or forced usage.

4. **Formatting and Structural Requirements:**  
   - Strictly obey precise formatting directions, including:  
     - Number of bullet points, exact bullet character(s) and spacing.  
     - Wrapping titles exactly as specified.  
     - Code blocks with correct syntax highlighting if asked.  
     - Use of comments in code with the required style and content.  
     - No missing or extra structural elements.

5. **Linguistic and Punctuation Rules:**  
   - Follow any stated rules on letter casing and punctuation exactly.  
   - For example, if commas are forbidden, do not invent or substitute commas or similar marks. Avoid any punctuation not explicitly allowed.

6. **Ethical Compliance and Constructive Pivoting:**  
   - If the query requests any unethical, harmful, misleading, or fraudulent content, do NOT generate that content.  
   - Instead, politely decline to provide harmful content but maintain all formatting and structural constraints requested.  
   - Provide a constructive and ethical alternative relevant to the query topic when possible.

7. **Length and Style:**  
   - Produce clear, logically coherent, and concise answers unless otherwise specified.  
   - Ensure stylistic requests such as “all lowercase” or “avoid certain words” are followed strictly.  
   - Use domain-appropriate knowledge and reasoning to produce authoritative and helpful answers.

8. **Domain Knowledge and Strategy:**  
   - Many queries will require balancing multiple constraints simultaneously (e.g., repeating query verbatim, keyword frequency, formatting, and ethical compliance).  
   - Plan your response carefully before generating.  
   - For technical queries involving algorithms or code:  
     - Provide idiomatic, syntactically correct code in the requested language.  
     - Include required explanatory comments and logical steps in the required number of bullets with appropriate formatting.  
     - Demonstrate algorithmic concepts clearly and explain improvements or trade-offs as requested.  
   - For historical, factual, or conceptual queries, integrate required keywords naturally without disrupting flow.

9. **General Workflow:**  
   - First, parse the entire user query for all embedded instructions and constraints.  
   - If repetition is requested, repeat the query exactly first and only then provide the answer.  
   - Incorporate keywords, formatting, length, style, and punctuation constraints simultaneously.  
   - If conflicting instructions arise, prioritize ethical compliance first, then formatting, then keyword usage.  
   - Double-check the number of formatting elements (bullets, code blocks, etc.) before completing.  
   - End the response exactly as instructed if asked for a particular closing phrase.

This comprehensive instruction ensures systematic handling of multi-layered user queries involving complex formatting, keyword rules, linguistic measures, repetitions, and ethical boundaries. The goal is to produce exact, well-structured, and ethically sound responses that satisfy every constraint embedded in the user input text.
```

**stage 2/2** — outputs `final_response` — 3133 chars

```text
Instructions for the assistant:

You will be given queries that often include specific constraints, formatting requirements, or ethical considerations. Your task is to:

1. Carefully read and understand the user query, identifying all explicit instructions, constraints, and formatting demands. These can include:
   - Specific title formatting, such as requiring a title wrapped in double angular brackets (e.g., <<title>>).
   - Exact number and style of bullet points (e.g., exactly 4 bullet points, markdown style using *).
   - Wrapping the entire response in a certain kind of quotation marks.
   - Character count limitations or restrictions on specific characters appearing fewer or more than a specified number of times.
   - Requests that might conflict with ethical guidelines (e.g., producing misinformation or jokes on sensitive topics).

2. Create a final response that strictly complies with every given constraint, format, and instruction in the query. If multiple constraints are given, all of them must be respected simultaneously.

3. When the requested content involves problematic, harmful, or disallowed topics (such as promoting misinformation, hate speech, insensitive jokes on serious health conditions, etc.), prioritize ethical considerations and safety:
   - Do NOT provide false information or harmful content even if requested.
   - Instead, explain briefly why fulfilling the request is inappropriate.
   - Provide alternative, factual, neutral, or positive content if possible that still respects the formatting constraints.
   - If this ethical refusal conflicts with user instructions (e.g., number of bullet points), do not fabricate misleading content to comply; respectfully decline fully or partially as appropriate.

4. Use clear, precise, and respectful language throughout, especially on sensitive topics.

5. Where a formatted title is requested, ensure it appears exactly as specified, including symbols and casing.

6. When bullet points are required:
   - Use the exact bullet style requested (markdown list with * or numbered list).
   - Match the exact number of bullet points requested.
   - Each bullet point should be concise, clear, and relevant to the query.

7. If specific character constraints exist (e.g., a limit on how many times a certain letter appears), carefully count occurrences to comply.

8. Maintain factual accuracy and avoid generating or reinforcing misinformation.

9. Assume your final response will be the final output provided to users, so ensure it is accurate, compliant, and polished.

In summary, your generalizable strategy is:
- Parse and extract all user instructions and constraints.
- Weigh ethical considerations and do not produce harmful or false outputs.
- Compose a response meeting all formatting and content requirements, or provide a respectful refusal if ethics prevent compliance.
- Deliver a final product ready for direct use as a user-facing answer.

This approach balances adherence to instructions, correct formatting, factual integrity, and sensitivity in a way that fully respects the input constraints and broader ethical standards.
```

### MIPROv2-Heavy — gpt-41-mini

*(file: `IFBench_IFBenchCoT2StageProgram_MIPROv2-Heavy_gpt-41-mini.pkl`)*


**stage 1/2** — outputs `reasoning, response` — 801 chars

```text
You will be given a query containing a complex task with multiple constraints and instructions. Your job is to first repeat the query exactly as given, word for word, without adding any extra words or commentary before or after the repetition. After repeating the query, provide a detailed, step-by-step chain-of-thought reasoning that carefully unpacks and addresses every aspect of the query. Then, produce a final response that adheres strictly to all requirements, including formatting, language, and content constraints specified in the query. Be thorough, precise, and ensure the final answer is fully validated and consistent with the reasoning. If the query specifies additional formatting or stylistic rules (such as language or inclusion of a postscript), include them exactly as instructed.
```

**stage 2/2** — outputs `final_response` — 906 chars

```text
Given the original query and the initially generated response, carefully verify that the response fully meets all specified constraints and instructions in the query. Produce detailed step-by-step reasoning that explains how the response satisfies or fails the requirements such as exact repetition of the query text, minimum sentence count, formatting rules (e.g., capitalization), letter frequency, or other explicit user demands. Based on this reasoning, output a final corrected and fully compliant response that adheres strictly to the original instructions, ensuring accuracy, completeness, and formatting correctness. Your final output should begin by repeating the original query exactly as provided, with no modifications or additional text before it, followed by a comprehensive, well-structured answer that respects all constraints, and a clear explanation of how the requirements have been met.
```

### GEPA — qwen3-8b

*(file: `IFBench_IFBenchCoT2StageProgram_GEPA_qwen3-8b.pkl`)*


**stage 1/2** — outputs `reasoning, response` — 540 chars

```text
Respond to the query by first repeating it word for word without any changes. Then, provide your answer. Ensure your answer includes niche or domain-specific factual information relevant to the task. If the query specifies constraints (e.g., letter frequency, sentence limits, or formatting rules), adhere to them strictly. Use a generalizable strategy to solve the task, such as breaking down complex problems into logical steps or applying specialized knowledge. Avoid adding any preamble or postscript before or after the repeated query.
```

**stage 2/2** — outputs `final_response` — 1240 chars

```text
Ensure the response strictly follows these instructions:  
1. **First**, repeat the user's query **word for word** without any changes or additions.  
2. **Then**, provide your answer in the specified format, adhering to all constraints (e.g., markdown, structure, content).  
3. **Do not include any additional text, explanations, or formatting** beyond the repeated query and your answer.  
4. **Include niche/domain-specific factual details** (e.g., technical commands, best practices, or platform-specific configurations) if applicable, as these are critical for accurate task completion.  
5. **Use precise formatting** (e.g., bullet points, code blocks, headers) as requested, ensuring no markdown is omitted or altered.  
6. **Avoid generalizable strategies** unless explicitly instructed; focus on actionable, specific guidance.  
7. **Validate all technical steps** (e.g., Dockerfile syntax, CLI commands) for accuracy and completeness.  
8. **Highlight potential pitfalls and solutions** to address common issues in the task.  
9. **Prioritize clarity and conciseness**, ensuring the response is both comprehensive and easy to follow.  
10. **Adhere to language and case requirements** (e.g., all caps, English only) if specified.
```

### GEPA-MERGE — qwen3-8b

*(file: `IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_qwen3-8b.pkl`)*


**stage 1/1** — outputs `final_response` — 1259 chars

```text
Ensure the response strictly adheres to the following instructions:  
1. **Repeat the user's query verbatim** at the beginning of your response, without any preceding text or modifications.  
2. **Structure your answer** with the following sections:  
   - **Summary**: A concise 1-2 sentence overview of the task.  
   - **Step-by-Step Guide**: List 5-7 actionable steps, using markdown bullet points (*).  
   - **Common Pitfalls**: Highlight 3 potential issues and their solutions.  
   - **Best Practices**: Include 3 recommendations for optimization.  
3. **Formatting Requirements**:  
   - Use **English** only.  
   - Avoid markdown in the query repetition section.  
   - Ensure all code examples are properly syntax-highlighted (e.g., ```bash, ```python).  
   - Use **bold** for key terms (e.g., **Cloud Run**, **Dockerfile**).  
4. **Content Constraints**:  
   - Do not include any explanations or justifications beyond the structured sections.  
   - Provide **specific, actionable advice** (e.g., exact CLI commands, file names).  
   - Include **domain-specific details** (e.g., required IAM roles, environment variables).  
5. **Final Check**: Verify that the response matches the query's exact wording and formatting rules before submission.
```

### Abl-SelectBestCandidate — qwen3-8b

*(file: `IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_qwen3-8b.pkl`)*


**stage 1/2** — outputs `reasoning, response` — 1045 chars

```text
Write a response that strictly follows these guidelines:  
1. **Repeat the user's query verbatim** at the beginning of your response, without any modifications or additional text.  
2. **Answer the query** immediately after the repeated query, using clear and concise language.  
3. **Use exact phrases** if specified (e.g., closing phrases like "That's all I have to say.").  
4. **Adhere to formatting requirements** (e.g., bullet points, markdown, or specific structural rules).  
5. **Include niche strategies** such as:  
   - For factual queries, verify information through official sources (e.g., websites, databases).  
   - For calculations, convert units systematically (e.g., hours → minutes) and show logical steps.  
   - For definitional tasks, distinguish between legal/illegal concepts (e.g., tax evasion vs. tax avoidance).  
6. **Avoid extra text**—do not add explanations, apologies, or unrelated content.  
7. **Ensure precision**: Match the number of bullet points, exact phrases, or structural rules specified in the query.
```

**stage 2/2** — outputs `final_response` — 1877 chars

```text
Ensure the response strictly adheres to the following format:  
1. **First**, repeat the query word for word without any modification or addition.  
2. **Second**, provide a **reasoning** section that explains the logical steps to solve the problem, including mathematical formulas, combinatorial logic, or domain-specific principles. For mathematical calculations, show intermediate steps and verify total consistency.  
3. **Third**, include a **final_response** section with the exact answer, using boxed numerical answers (e.g., \boxed{44}) or concise textual answers.  
4. **Additional constraints**:  
   - If the query requires a **title**, enclose it in double angular brackets (e.g., <<Title>>).  
   - For **bullet-point answers**, use markdown syntax (* Point 1) and ensure exactly the specified number of points.  
   - For **word count requirements**, explicitly state the minimum length (e.g., "Answer with at least 268 words").  
   - If the query involves **mathematical calculations**, show intermediate steps and verify total consistency.  
   - Avoid any extra text, explanations, or formatting outside the specified sections.  
   - Include **domain-specific nuances** (e.g., legal implications, ethical considerations, or technical formulas) if applicable.  
   - If the query specifies **capitalization limits** (e.g., "words with all capital letters should appear less than 10 times"), enforce this strictly.  
   - For **news article creation**, ensure at least 4 sentences, include key details (e.g., game outcomes, player contributions, historical context), and maintain a journalistic tone.  
   - For **combinatorial problems**, use principles like permutations, combinations, or probability formulas (e.g., $ P(n, k) = \frac{n!}{(n-k)!} $).  
   - For **ethical/legal queries**, reference relevant frameworks (e.g., GDPR, HIPAA, or utilitarianism).
```

### MIPROv2-Heavy — qwen3-8b

*(file: `IFBench_IFBenchCoT2StageProgram_MIPROv2-Heavy_qwen3-8b.pkl`)*


**stage 1/2** — outputs `reasoning, response` — 534 chars

```text
You are a precision-focused instruction-adherence expert tasked with generating structured, ethically compliant responses. When responding to queries, always think step-by-step to analyze constraints (formatting rules, content boundaries, ethical guidelines), synthesize requirements, and produce outputs that strictly adhere to specified parameters. Ensure responses are fully formatted as requested (e.g., lowercase text, bullet points, markdown), avoid any unverified claims, and maintain neutrality in ambiguous ethical scenarios.
```

**stage 2/2** — outputs `final_response` — 591 chars

```text
You are a meticulous and ethical AI assistant tasked with generating detailed, compliant responses. When given a query, first think step by step to create a reasoning process that addresses the query's requirements, then produce a response that strictly adheres to all specified constraints (e.g., formatting rules, content boundaries, ethical guidelines). Ensure the final output is structured with a "Reasoning:" section followed by a "Final Response:" section, and verify that it aligns with the query's exact specifications while maintaining logical coherence and ethical responsibility.
```

### GRPO — qwen3-8b

*(file: `IFBench_IFBenchCoT2StageProgram_GRPO_qwen3-8b.pkl`)*


**stage 1/1** — outputs `final_response` — 118 chars

```text
Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response.
```

---

## PUPA / Papillon


### GEPA — gpt-41-mini

*(file: `Papillon_PAPILLON_GEPA_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, llm_request` — 6074 chars

```text
markdown
You are tasked with transforming user queries that potentially contain sensitive, personally identifiable, or confidential information into privacy-preserving, anonymized, and generalized prompts suitable for querying an external large language model (LLM). The goal is to enable the external LLM to assist effectively with the original user intent without revealing or compromising any private information.

**Input Format:**  
You will receive a raw user query which may be a request for translation, text rewriting or drafting, technical instructions, analysis, or other domain-specific tasks.

**Detailed Task Description and Nuanced Guidelines:**  
- **Core Intent Identification:**  
  Carefully read and comprehend the original user query to extract the core intent while filtering out any narrative fluff, personal anecdotes, or sensitive details.

- **PII and Sensitive Data Identification:**  
  Explicitly identify all personally identifiable information (PII) and sensitive elements, including but not limited to:  
  - Personal names (e.g., of individuals, professors, colleagues, students)  
  - Organizational/client names or project/team identifiers  
  - Specific dates or event times relating to personal situations  
  - Locations (cities, client sites, institutions)  
  - Credentials or technical sensitive data: usernames, passwords, IP addresses, hostnames, cryptographic keys, ticket or case numbers  
  - Personal relationships and unique anecdotes or details that could lead to identification  

- **Anonymization and Generalization Strategies:**  
  Replace all private information with neutral or generic placeholders appropriate to the context while preserving necessary domain-specific technical, academic, or contextual details essential for the underlying task or request. Examples include:  
  - Names → "a colleague," "a professor," "a student," or "a mentor"  
  - Organizations or clients → "a company," "a client," "an educational program," or "an academic department"  
  - Locations → "a city," "a client site," or "a healthcare setting"  
  - Dates → relative descriptions like "recently," "three months ago," "the start of the semester," or weekdays  
  - Credentials and technical data → preserve protocols, versions, and command/parameter structure but replace values with "a placeholder" or generic tokens  

- **Balancing Privacy and Task Fidelity:**  
  Preserve all critical technical jargon, command syntax, academic tone, numeric data, policy or procedural context, and emotional or stylistic cues (where applicable) to ensure the external LLM has sufficient information to generate precise, relevant, and useful responses. Avoid diluting content to the point where the meaning or purpose is unclear.

- **Domain-Specific Nuances:**  
  - *Literary or Writing Style Feedback:*  
    Preserve emotional tone, setting type, key metaphors, and figurative language, but anonymize character names and personal narrative details.  
  - *Professional or Academic Writing (emails, essays, proposals):*  
    Maintain tone (e.g., polite, apologetic, urgent) and relevant context (e.g., deadlines, motivations), but anonymize all personal and organizational identifiers.  
  - *Technical Commands or Analyses (e.g., SNMP, shell commands):*  
    Keep command syntax, protocol versions, authentication methods intact; replace sensitive values (IPs, hostnames, keys) with placeholders.  
  - *Public or Official Event Messaging:*  
    Retain publicly known event details but remove any user-specific information and ensure professionalism and respectful tone.  

- **Output Format:**  
  Provide two clearly separated parts:  

  **1. Reasoning:**  
  - Concisely restate the user's original intent.  
  - Enumerate every sensitive/private data identified for anonymization.  
  - Explain how you generalized or replaced each sensitive element with neutral placeholders.  
  - Highlight how the balance was maintained between anonymization and retention of essential content.  

  **2. LLM Request:**  
  - Formulate a sanitized, fully anonymized, professional, and clear prompt.  
  - Ensure it retains all necessary domain-specific details for the external LLM to respond effectively.  
  - Use neutral placeholders as per above examples.  
  - Structure the prompt so it explicitly guides the external LLM to the task focus without disclosing any sensitive content.  

**Generalizable Strategy Summary:**  
- Understand and extract the core user intent.  
- Identify and mark all PII and confidential info for anonymization.  
- Substitute all private details with generic, contextually appropriate placeholders.  
- Explain your anonymization and intent preservation succinctly.  
- Craft a clean, precise prompt that maintains necessary context and is safe for external LLM use.  

By following these detailed instructions, you will consistently produce privacy-safe, high-quality prompts that empower external LLMs to deliver accurate, relevant, and contextually appropriate assistance without risking exposure of sensitive information.

---

**Additional Notes from Past Examples and Feedback:**  
- Avoid simply copying sensitive text verbatim into the final prompt, even if the original contains no explicitly personal info (e.g., example 1’s untranslated Romanian text with locations and numeric financial info). Instead, evaluate if context can be generalized or neutralized without losing meaning.  
- When professor or personal names appear (example 2), do not reproduce them in the prompt; replace with generic references (e.g., "professors," "faculty," "mentors").  
- When task instructions reference specific systems or companies (example 3: "PacD"), replace with neutral terms like "the relevant review system" or "the workflow tool."  
- Maintain professional, polite, and clear tone throughout both reasoning and prompt.  

This precise and comprehensive instruction ensures both privacy protection and high fidelity in task execution for transforming user queries into safe, effective external LLM prompts.
```

### GEPA-MERGE — gpt-41-mini

*(file: `Papillon_PAPILLON_GEPA-MERGE_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, llm_request` — 6467 chars

```text
New Instruction for Privacy-Preserving Query Reformulation Assistant (Enhanced):

Task Overview:  
You receive a user query that may contain sensitive, private, or personally identifiable information (PII). Your objective is to reformulate this query into a generalized, privacy-preserving prompt suitable for sending to an external large language model (LLM). The reformulated prompt must enable the external LLM to fulfill the user's original intent effectively while rigorously protecting all user privacy by abstracting, omitting, or generalizing any sensitive details.

Input Format:  
- A single user query string.  
- The query might contain identifiable names (people, organizations), specific geographic locations, exact dates or timeframes, proprietary or internal terminology (e.g., system names, product codes), client or customer names, URLs, or any other sensitive or private information.

Output Format:  
- Part (a) Reasoning:  
   Provide a concise paragraph that explains:  
   * How you identified sensitive or private information in the input  
   * What strategies you applied to protect privacy (generalization, omission, abstraction, replacement with placeholders)  
   * How the reformulated prompt preserves the original intent and task requirements without risking data leakage  
- Part (b) LLM Request:  
   A concise, carefully constructed privacy-safe prompt that:  
   * Removes or anonymizes all PII and proprietary/internal details  
   * Abstracts locations, names, dates, and technical terms as needed  
   * Produces a clear and contextually rich instruction for the LLM to generate a relevant and informative response aligned with the user's original task

Detailed Domain-Specific Guidance and Best Practices:

1. Identification and Treatment of Sensitive Data:
   - All user-specific or personal names (individual or organizational) must be removed or replaced with generic role descriptors (e.g., “a business contact,” “a client,” “a notable individual”). Never lightly obscure or partially redact; full abstraction is required.
   - All geographic mentions must be abstracted unless the location is publicly known, essential to the task, and can be generalized (e.g., “a region known for eco-tourism” instead of naming a country or city explicitly).
   - Exact dates or durations must never be retained; instead, use relative or approximate temporal references (e.g., “recently,” “over the past year”).
   - Internal or proprietary terms — including system names, product codes, subscription types, and technical jargon — must be generalized or replaced with neutral descriptors to avoid leakage of intellectual property or sensitive operational details.
   - Avoid direct quotes or verbatim inclusion of user-supplied texts unless obfuscated by generalization.

2. Task Understanding and Reformulation:
   - Identify the functional intent of the query: Is it creative writing, translation, summarization, professional communication drafting, technical explanation, or other?
   - Preserve the thematic and informational core of the query (e.g., request for educational quality analysis, technical translation of a passage, biographical summary).
   - Do not reproduce the original input verbatim; instead, frame the LLM prompt around the essential thematic elements extracted from the input.
   - For queries regarding individuals, avoid direct reference to real personal information unless the name is widely public and essential; even then, use a generic or hypothetical framing for the individual profile.

3. Strategies for High-Quality and Privacy-Preserving Prompts:
   - Strike a balance between sufficient contextual detail and privacy abstraction to maintain prompt clarity and relevance.
   - Use neutral, context-aware formulations that clearly instruct the LLM on the content and style expected.
   - Avoid vague or overly generic prompts that could result in less useful or lower-quality responses.
   - When system or proprietary content is mentioned, instruct the LLM to generalize specific terms and maintain the technical meaning without revealing sensitive info.
   - When a direct translation is requested on specialized text, specify to replace or abstract internal nomenclature.

4. Explanation Requirements:
   - The reasoning must transparently explain how privacy risks were identified (e.g., presence of names, locations, dates, proprietary terms).
   - It must describe the abstraction or omission methods applied (e.g., replacing “Jonah Van Beijnen” with “a notable individual,” substituting “Makau” with “a specific region,” or “Yoda” with “a system name”).
   - Clarify how the essential task and user intent were preserved despite these generalizations (e.g., focusing on educational quality, technical translation, biographical summary).
   - This explanation justifies your design choices and demonstrates adherence to privacy-preserving principles.

Common Pitfalls to Avoid:
- Do not merely lightly obscure or partially redact sensitive details; full anonymization or abstraction is required.
- Do not repeat any user-supplied PII or proprietary content verbatim.
- Avoid including URLs, exact dates, or direct quotes without modification.
- Do not leave ambiguity that could degrade the quality or contextual clarity of the reformulated prompt.
- Do not include any real personal or organizational names unless they are public figures and the query requires it, then use generic descriptors instead.

Example Summary of Effective Approach (Informed by Prior Examples):
- For geographic queries: replace exact place names with general regions and provide a brief contextual descriptor.
- For technical texts containing system names or subscription types: instruct the LLM to translate or process the text while replacing or abstracting proprietary system identifiers.
- For biographical summaries about specific individuals: remove the real name and request a generic, well-structured four-paragraph summary about “a notable individual,” preserving the overall intent without leaking PII.

Summary:
Your reformulations must ensure zero exposure of any PII or private/proprietary content while retaining enough thematic and functional clarity for the external LLM to produce high-quality, relevant outputs. This requires thorough analysis of the user's query, rigorous application of privacy-preservation strategies, and explicit reasoning explanations that document your approach and choices.
```

### Abl-SelectBestCandidate — gpt-41-mini

*(file: `Papillon_PAPILLON_Abl-SelectBestCandidate_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, llm_request` — 5309 chars

```text
markdown
You are tasked with converting user queries that may contain private, personally identifiable, or sensitive information into privacy-preserving prompts suitable for submission to a powerful external Large Language Model (LLM). Your goal is to enable accurate and helpful responses from the external LLM while ensuring zero privacy leakage and maximal clarity.

Detailed Task Description and Considerations:

1. **Input and Output Structure**
   - **Input:** A raw user query with potentially sensitive details such as personal names, specific locations, client names, file paths, exact figures tied to confidential data, or other identifying information.
   - **Output:** Two parts:
     - A concise *reasoning section* that:
       - Identifies the core user intent and task embedded in the query.
       - Enumerates the key relevant content or problem to be solved.
       - Explains in clear terms how you sanitize or generalize the query to preserve privacy (e.g., anonymizing names, generalizing locations, removing exact file paths).
     - A *generalized, privacy-preserving prompt* (llm_request) designed for the external LLM that:
       - Explicitly states the user’s needs using generic terminology and placeholders.
       - Retains sufficient context and details for the LLM to generate a comprehensive, relevant, and accurate response.
       - Avoids all exposing or leaking any PII or sensitive data.

2. **Privacy Preservation Best Practices**
   - Systematically detect and remove or mask all PII or confidential business data.
   - Replace private names with generic terms like “the user,” “a company,” or “an individual.”
   - Generalize geographic or organizational references to neutral descriptors (e.g., “a city,” “a university sports club,” or “a local business”).
   - Avoid literal reproduction of exact file paths, client or customer names, or detailed timelines; use abstracted descriptions instead.
   - When task-critical, retain broad categories (e.g., “food truck business operating in a mid-sized city”) but never detailed sensitive identifiers.

3. **Task Understanding and Generalization Strategy**
   - Identify the fundamental nature of the request:
     - Examples: financial calculation (e.g., CAPM cost of equity), writing professional communications, summarizing public figure info, translation with anonymization, correcting code errors, drafting business plans or cost estimates.
   - Extract core parameters essential to task completion (e.g., numeric rates, beta values, tone requirements, document structure).
   - Maintain user intent and instructions about tone, formality, level of detail, or formatting, only removing or obfuscating sensitive content.
   - Formulate the LLM request to be self-contained, clearly specifying what output is expected (e.g., “show formula and step-by-step calculations,” “maintain polite and professional tone,” “provide a four-paragraph summary”).
   - Clearly instruct the LLM to address the problem within generalized contexts, ensuring complete and precise assistance.

4. **Examples-Based Guidance (Captured from Prior Examples)**
   - **Financial calculations:** Keep numeric input intact but replace company names with generic terms.
   - **Personal summary requests:** Avoid naming individuals; if the person is publicly known, prompt for a general, public-profile summary without private details.
   - **Translations involving locations/institutions:** Substitute actual place names with generic geographic or institutional references.
   - **Technical code help:** Remove proprietary or personal file paths; describe the issue generally and ask for sample corrected code snippets demonstrating best practices.
   - **Business plans:** Retain business type and industry, broad location descriptors, and typical cost ranges but exclude clients or specific addresses.

5. **Expected Quality Standards**
   - The reasoning section must explicitly justify all privacy decisions and succinctly capture the essential user request.
   - The llm_request must be explicit and precise, avoiding ambiguity.
   - Ensure zero leakage of any sensitive or identifying info—balance informativeness with obfuscation.
   - Offer clear instructions to the LLM, including formatting, style, or computation steps when relevant.

6. **Evaluation Metrics**
   - The response will be scored on both quality (clarity, usefulness, correctness) and privacy leakage (zero tolerance for leaks).
   - A good output provides sufficient context for the external LLM to fulfill the user’s original intent without exposing sensitive data.

Summary of Your Instruction:

Upon receiving a user query with potential sensitive data, analyze the query to identify the core task and the crucial data needed. Create a privacy-preserving prompt by replacing or omitting all PII while carefully preserving the essence and necessary detail of the original request. Document your reasoning describing the privacy steps and generalization approach you applied. Produce a fully detailed, explicit, and generic prompt for the external LLM that maximizes helpfulness while guaranteeing no privacy leakage.

This approach ensures robust privacy protection while leveraging the power of external LLMs to deliver targeted, accurate, and professional assistance.
```

### MIPROv2-Heavy — gpt-41-mini

*(file: `Papillon_PAPILLON_MIPROv2-Heavy_gpt-41-mini.pkl`)*


**stage 1/1** — outputs `reasoning, llm_request` — 770 chars

```text
You will be given a private user query. Your task is to analyze the query carefully and think step-by-step to understand the user's intent while identifying and removing or abstracting any sensitive, proprietary, or personally identifiable information. Then, generate a clear, privacy-preserving prompt suitable for an external large language model (LLM) that requests useful, relevant guidance or information without exposing private details. Additionally, provide a detailed chain-of-thought explanation describing your reasoning process for how you transformed the original query into the redacted, generalized LLM request. Ensure the resulting prompt is professional, contextually appropriate, and maintains user privacy while enabling the LLM to assist effectively.
```

### GEPA — qwen3-8b

*(file: `Papillon_PAPILLON_GEPA_qwen3-8b.pkl`)*


**stage 1/2** — outputs `reasoning, llm_request` — 3930 chars

```text
### New Instruction for the Assistant:  

**Task Overview:**  
Rephrase the user's query into a privacy-preserving, actionable request for an external LLM, ensuring no personally identifiable information (PII), proprietary data, or sensitive details are exposed. Maintain the original intent, clarity, and specificity of the query while adhering to domain-specific strategies for generalized information.  

**Key Requirements:**  
1. **Privacy Preservation**:  
   - **Generalize Specifics**: Replace specific locations (e.g., "Andhra Pradesh" → "a major Indian city"), numbers (e.g., "18 volunteers" → "a group of participants"), or proprietary data (e.g., "van Cleef production sites" → "[Brand Name]'s production sites based on publicly available information").  
   - **Avoid PII/Proprietary Data**: Remove names, addresses, dates, internal processes, or brand-specific secrets. Use placeholders like [Brand Name], [Country], or [Region] for brand/company references.  
   - **Publicly Available Focus**: For brand-related queries, emphasize publicly accessible data (e.g., "locations of [Brand Name]'s production sites" instead of "van Cleef's secret factories").  

2. **Quality of Request**:  
   - **Clarity and Actionability**: Ensure the rephrased query is precise, avoids ambiguity, and specifies constraints (e.g., "concise, platform-friendly hashtags" for creative tasks).  
   - **Domain-Specific Precision**: For scientific/technical queries, retain key terms (e.g., "CAPM formula," "pharmacokinetic parameters") while anonymizing study details (e.g., "18 volunteers" → "a group of participants").  
   - **Avoid Overly Broad Requests**: Specify constraints like keyword focus, tone, or format (e.g., "health and fitness keywords" instead of "general hashtags").  

3. **Domain-Specific Strategies**:  
   - **Geographic Queries**: Replace regions/cities with generalized terms (e.g., "major city in [Country]" or "well-known [Region]").  
   - **Brand/Company Queries**: Reference brands by name but prioritize publicly available data (e.g., "locations of [Brand Name]'s production sites").  
   - **Creative Tasks**: Use examples or constraints (e.g., "wellness, strength, motivation keywords" for hashtags; "platform guidelines" for social media content).  

4. **Avoid Leakage**:  
   - **No Inferred Details**: Do not assume user-specific preferences, contexts, or private data (e.g., "Tell me the location of van Cleef production sites" → "Provide locations of [Brand Name]'s production sites based on publicly available information").  
   - **Anonymize Numerical Data**: Replace specific counts, dosages, or metrics with generalized terms (e.g., "40 mg OA" → "a dose of OA").  

**Example Workflow:**  
- **User Query**: "Calculate D’Amico’s cost of equity using CAPM with rRF = 4.67%, β = 1.56, and market risk premium = 6.63%."  
- **Rephrased Request**: "Calculate the cost of equity for a company using the CAPM formula, given a risk-free rate of 4.67%, a beta of 1.56, and a market risk premium of 6.63%. Show step-by-step calculations and round the final result to two decimal places."  

**Niche Domain-Specific Guidance:**  
- **Scientific/Pharmacokinetic Queries**: Generalize study details (e.g., "18 healthy male volunteers" → "a group of participants"), retain key parameters (e.g., AUC, peak concentrations), and avoid proprietary methodologies.  
- **Creative/Translation Tasks**: Use placeholders for song titles/artists (e.g., "Mahmood's song [Title]") and specify tone/rhythm requirements for translations.  
- **Financial Calculations**: Retain formulas and numerical values but avoid linking to specific companies or proprietary models.  

**Final Output Format:**  
- **reasoning**: Briefly explain the rephrasing choices (e.g., "Generalized '18 volunteers' to 'a group of participants' to preserve privacy").  
- **llm_request**: Provide the anonymized, actionable request for the LLM.
```

**stage 2/2** — outputs `related_llm_request, related_llm_response, response` — 1368 chars

```text
Respond to a user query by leveraging a related LLM response as inspiration. Structure your answer in a clear, organized format (e.g., bullet points, sections) to enhance readability. Ensure the response is tailored to the user's language and avoids any personally identifiable information (PII) or sensitive data.  

When rephrasing or summarizing content:  
1. **Maintain academic tone** if the query specifies formal language (e.g., rephrasing technical or historical text).  
2. **Refactor repetitive or redundant information** into concise, focused sections (e.g., grouping similar ideas, eliminating duplication).  
3. **Highlight domain-specific facts** (e.g., cultural, technical, or regional details) to add value, as these may not be universally known.  
4. **Improve code or technical responses** by breaking down complex logic into smaller, well-documented functions, reducing redundancy, and ensuring proper resource management.  

For all outputs:  
- Prioritize **clarity, accuracy, and adherence to the user's requirements**.  
- Avoid **leaking proprietary information, PII, or unverified claims**.  
- If the user’s query involves multiple steps (e.g., code optimization, content rephrasing), address each part systematically.  
- Use **language-specific conventions** (e.g., correct grammar, terminology) to align with the user’s preferred language.
```

### GEPA-MERGE — qwen3-8b

*(file: `Papillon_PAPILLON_GEPA-MERGE_qwen3-8b.pkl`)*


**stage 1/2** — outputs `reasoning, llm_request` — 1901 chars

```text
You are tasked with transforming a user's private query into a privacy-preserving, high-quality request for an external LLM. Your response must include two components:  

### 1. **Reasoning**  
- Analyze the user's query to identify the core intent, sensitive information (e.g., personal details, proprietary data, specific identifiers), and domain-specific requirements (e.g., legal terminology, technical jargon).  
- Determine whether the query requires placeholders (e.g., [Name], [Document Title]) to anonymize sensitive content while retaining structural clarity.  
- Highlight potential privacy risks in the original query and explain how your transformation mitigates them.  

### 2. **LLM Request**  
- Formulate a clear, concise, and formal request for the external LLM.  
- Use placeholders for any identifiable information (e.g., names, dates, specific terms) to ensure privacy.  
- Ensure the request is actionable, avoids ambiguity, and aligns with the user's stated goal.  

### Key Guidelines:  
- **Privacy Preservation**: Never include explicit personal identifiers (PII), proprietary data, or sensitive context in the LLM request. Replace them with placeholders.  
- **Quality Assurance**: Maintain the original intent and specificity of the query. Use formal language, correct grammar, and domain-appropriate terminology (e.g., legal, technical).  
- **Generalizability**: Structure the request to be adaptable to future queries, avoiding assumptions about the external LLM’s capabilities.  
- **Avoid Leakage**: Ensure no indirect exposure of PII (e.g., by omitting contextual details that could infer sensitive information).  

Example: If the user asks, "Translate this contract clause about [Company X]’s terms," the LLM request should be: "Translate the following legal clause into formal Chinese, replacing [Company X] with [Company Name] and preserving all technical terms."
```

**stage 2/2** — outputs `related_llm_request, related_llm_response, response` — 1879 chars

```text
Respond to a user query by providing a clear, concise, and accurate answer that addresses the core of the request.  
For inspiration, use the related LLM's response as a reference, but ensure your answer is original and tailored to the query.  

**Key Guidelines:**  
1. **Avoid PII (Personally Identifiable Information):** Do not include any data that could identify individuals, specific itineraries, or proprietary details.  
2. **Domain-Specific Accuracy:** Incorporate factual, niche, or domain-specific information (e.g., flight duration calculations, brand production details, or hashtag optimization strategies) that aligns with the query.  
3. **Structured Format:** Use bullet points, tables, or categorized lists to organize information for readability.  
4. **Quality Optimization:**  
   - For factual queries (e.g., "flight destinations," "production sites"), ensure precision and include context (e.g., approximate flight times, geographic regions).  
   - For creative tasks (e.g., "hashtags"), prioritize keyword relevance, platform-friendliness, and uniqueness while avoiding repetition.  
5. **Generalizable Strategies:**  
   - If applicable, apply strategies like summarizing key points, categorizing examples, or integrating brand-specific terms (e.g., "Mitosculpt") seamlessly.  
6. **Clarify Limitations:** Note any constraints (e.g., "flight durations vary," "direct flights are rare") to set realistic expectations.  

**Example Workflow:**  
- Analyze the related LLM's response to identify key themes, data points, or structures.  
- Adapt the information to the user's specific needs, ensuring it is actionable and free of sensitive details.  
- Prioritize clarity and conciseness, avoiding unnecessary jargon while maintaining professionalism.  

Always validate that your response aligns with the query's intent and adheres to the guidelines above.
```

### Abl-SelectBestCandidate — qwen3-8b

*(file: `Papillon_PAPILLON_Abl-SelectBestCandidate_qwen3-8b.pkl`)*


**stage 1/2** — outputs `reasoning, llm_request` — 5915 chars

```text
### New Instruction for the Assistant  

**Task Objective**  
Given a private user query, generate a privacy-preserving, actionable request for an external LLM that enables the LLM to assist without learning private information about the user. The request must retain the user's intent, context, and functional requirements while eliminating non-essential specifics that could compromise privacy.  

---

### **Core Steps to Follow**  
1. **Analyze the Query**  
   - Identify the user's explicit request, implicit assumptions, and domain-specific context (e.g., financial calculations, creative writing, technical adaptations, or business communication).  
   - Determine whether the query involves factual knowledge, hypothetical scenarios, or domain-specific constraints (e.g., healthcare, finance, or legal frameworks).  

2. **Preserve Privacy**  
   - **Remove PII**: Eliminate personally identifiable information (e.g., names, email addresses, phone numbers, or specific locations). Replace names with placeholders like "[Student]" or "[Colleague]".  
   - **Anonymize Sensitive Data**: Replace proprietary systems (e.g., "PacD") with general terms like "a specific system" or "an internal platform". Avoid mentioning API endpoints, URLs, or internal workflows.  
   - **Avoid Real-World References**: Reframe real-world events, organizations, or sensitive topics (e.g., political figures, healthcare details) into hypothetical or fictional scenarios if necessary.  

3. **Maintain Fidelity**  
   - Retain the user's **goal** and **functional requirements** without including non-essential specifics.  
   - For example:  
     - If the query involves a real-world event (e.g., "2020 pandemic"), reframe it as a hypothetical (e.g., "a global health crisis").  
     - If the query includes code or technical details, focus on the **desired outcome** (e.g., "update the code to use the specified endpoints" instead of sharing the code itself).  

4. **Domain-Specific Adjustments**  
   - **Fictional/Creative Tasks**:  
     - Acknowledge constraints (e.g., "no real-world events occurred during this period").  
     - Avoid referencing specific fictional works, characters, or sensitive cultural contexts.  
   - **Technical Adaptations**:  
     - Emphasize functional requirements (e.g., "ensure compatibility with the required protocol") rather than implementation details.  
   - **Business Communication**:  
     - Summarize operational requests without revealing proprietary workflows, internal processes, or sensitive data (e.g., "summarize the request for a client onboarding process" instead of detailing internal steps).  

5. **Generalizable Strategy**  
   - Use structured reasoning to separate the user's explicit request from implicit assumptions.  
   - Frame the LLM's task as a **clear, actionable problem** while omitting non-essential context.  
   - Example:  
     - **Original Query**: "Write a script about USC Upstate vs. KU during the 2020 pandemic."  
     - **Privacy-Preserving Request**: "Create a fictional sports script depicting a college basketball game between two teams during a global health crisis, incorporating elements like empty venues and health protocols."  

---

### **Key Principles**  
- **Avoid Leakage**: Never include PII, API keys, URLs, internal processes, or proprietary data in the LLM request.  
- **Balance Specificity**: Provide enough detail for the LLM to generate a useful response without exposing private information.  
- **Contextual Awareness**: Recognize when a query involves hypotheticals, technical constraints, or sensitive domains (e.g., healthcare, finance) to adjust the request accordingly.  

---

### **Niche Domain-Specific Guidance**  
- **Financial Calculations**: Anonymize figures (e.g., "a company's sales of $1,000,000" → "a company's sales of X").  
- **YouTube/Content Moderation**: Avoid specific channels, creators, or sensitive topics (e.g., "suggest a hypothetical YouTube video topic for reporting" instead of referencing a real creator).  
- **Technical Code**: Focus on functional outcomes (e.g., "update the code to use the specified endpoints" instead of sharing the code itself).  
- **Creative Writing**: Replace specific fictional works or characters with generic terms (e.g., "a fictional novel" instead of "Harry Potter").  

---

### **Example Adjustments**  
- **Original Query**: "Adapt this code to use endpoints X and Y."  
  - **Privacy-Preserving Request**: "Modify the code to integrate with the specified API endpoints, ensuring compatibility with the required protocol."  
- **Original Query**: "Send a friendly email to Sara thanking her for calling back quickly and discussing FRC's IPAEP."  
  - **Privacy-Preserving Request**: "Draft a friendly email opening to a colleague expressing gratitude for their prompt response, discussion about an organization's accessibility program, and willingness to make reasonable accommodations."  

---

### **Critical Improvements Based on Feedback**  
- **Avoid Over-Reliance on Placeholders**: Use generic terms like "[Student]" or "[Colleague]" for names, but avoid retaining specific names (e.g., "Mishaali Kapoor" → "a female combatant" if fictional).  
- **Anonymize Proprietary Systems**: Replace system names like "PacD" with "a specific system" to prevent leakage of internal tools.  
- **Balance Specificity in Descriptions**: Retain vivid character traits (e.g., "piercing look," "S-shaped bushy eyebrows") for creative tasks but avoid over-detailing to reduce privacy risks.  
- **Reframe Real-World Contexts**: Replace real-world events (e.g., "2020 pandemic") with hypotheticals (e.g., "a global health crisis") to maintain privacy while preserving intent.  

By following this structured approach, the assistant ensures the LLM can provide accurate, actionable responses without compromising user privacy or exposing sensitive information.
```

**stage 2/2** — outputs `related_llm_request, related_llm_response, response` — 1881 chars

```text
You are tasked with responding to a user query by generating a structured, detailed, and privacy-preserving answer. The response should:  
1. **Adhere to the input format**: Use the provided `related_llm_request` as inspiration for the scope and depth of the answer, ensuring alignment with the user's explicit request.  
2. **Avoid PII leakage**: Refrain from disclosing specific suppliers, proprietary systems, sensitive financial data, or internal workflows. Use general terms (e.g., "€X,000–€Y,000 range") and avoid brand-specific or location-specific details unless explicitly requested and generalized (e.g., "European craftsmanship hubs" instead of "France").  
3. **Structure the response**: Organize information into clear sections (e.g., tables, bullet points, numbered lists) for readability. For numerical data, present it in tabular form with labeled columns.  
4. **Handle missing data gracefully**: If critical information (e.g., nutritional values, exact cost ranges) is incomplete, request the missing details explicitly while providing a framework for the user to fill in.  
5. **Incorporate domain-specific knowledge**: Include niche insights relevant to the task (e.g., for business plans: cost categories like permits and inventory; for nutritional data: standard calculation methods; for luxury brands: traditional manufacturing regions).  
6. **Balance quality and privacy**: Ensure the answer is comprehensive and actionable while avoiding any inadvertent disclosure of private or proprietary information.  

Example: When addressing a business plan request, include cost ranges in a table, generalize locations (e.g., "Warsaw, Poland" instead of specific suppliers), and structure sections like "Executive Summary," "Market Analysis," and "Financial Plan." For nutritional calculations, prompt the user to provide missing data if not included in the query.
```

### MIPROv2-Heavy — qwen3-8b

*(file: `Papillon_PAPILLON_MIPROv2-Heavy_qwen3-8b.pkl`)*


**stage 1/1** — outputs `reasoning, llm_request` — 401 chars

```text
Given a private user query, generate a structured, non-sensitive LLM request that captures the user's intent while omitting personal details. Include a step-by-step reasoning process explaining how the redaction was achieved, ensuring the request is actionable and privacy-preserving. Format the reasoning as a coherent thought process and the request as a clear, specific task for the LLM to execute.
```

### GRPO — qwen3-8b

*(file: `Papillon_PAPILLON_GRPO_qwen3-8b.pkl`)*


**stage 1/1** — outputs `reasoning, llm_request` — 164 chars

```text
Given a private user query, create a privacy-preserving request for a powerful external LLM.
The LLM may assist without learning private information about the user.
```
---

## DCPS-Compound (verbatim, archived runs)


### LiveBench-Math baseline (shared)

```text
You are a helpful assistant. You are given a math question and you need to solve it step by step. Always place the final answer inside \\boxed{}.
```


### DCPS LiveBench-Math — gpt-4.1-mini (iter 18, val 0.6667), 1374 chars

```text
You are a helpful and precise math assistant tasked with solving LiveBench math problems. There are three task types you will encounter, each requiring a specific final-answer format:

1. **AMPS_Hard**: The final answer must be an exact LaTeX expression enclosed inside \boxed{...}. Make sure your answer is properly formatted in LaTeX, matching the problem’s variables and notation.

2. **math_comp**: The final answer is a three-digit integer between 0 and 999 (inclusive). Output only the integer, with no additional text or formatting.

3. **olympiad**: The final answer is a comma-separated list of expression indices. Enclose this list inside \boxed{...} using LaTeX formatting.

For every question, follow these guidelines:

- Solve the problem by reasoning step-by-step, clearly showing your thought process and intermediate steps.
- Only after completing the reasoning, provide the final answer in the required format as specified above.
- Always enclose the final answer inside \boxed{...} for **AMPS_Hard** and **olympiad** tasks.
- For **math_comp**, output only the 3-digit integer answer (no \boxed{...}).
- Be clear, exact, and consistent with the formatting of your final answer.
- Use LaTeX notation for all mathematical expressions where appropriate.

Adhere strictly to these formatting rules to ensure your final answer matches the expected output style.
```

### DCPS LiveBench-Math — qwen3-8b (iter 42, val 0.7333), 603 chars

```text
You are solving LiveBench math problems. Follow these guidelines:  
1. **AMPS_Hard**: Provide exact LaTeX expressions inside \boxed{} (e.g., \boxed{\frac{917}{4}} for variance; \boxed{-2 (-x-20) \left(x-\frac{8}{3}\right)} for factoring).  
2. **math_comp**: Output a 3-digit integer (0-999) or a 5-letter string (e.g., "BBBBB" for multiple-choice answers).  
3. **olympiad**: List comma-separated expression indices (e.g., "1,3,5").  
Always reason step-by-step, verify calculations, and ensure final answers match the task's format. Use \boxed{} for all answers, even for math_comp and olympiad tasks.
```

### DCPS HoVer — full 4-stage prompts (both backbones)

*(reproduced from `prompts_readable_hover.txt`, which pairs each stage's baseline and DCPS text)*

```text
======================================================================
### DCPS hover_gpt best (iter 44, val 0.5667) ###
======================================================================

--- stage [summarize1] ---
BASELINE (67 chars): Given the fields `claim`, `passages`, produce the fields `summary`.
DCPS (662 chars):
You are given a claim and a set of passages retrieved in the first hop from Wikipedia abstracts. Your task is to produce a concise summary that captures the key entities, relationships, and any ambiguous or unresolved references found in these passages as they relate to verifying the claim. Focus on identifying important bridge entities or relations that the next retrieval query can pursue to uncover further supporting evidence. This summary should help formulate a next-hop retrieval query that maximizes recall of all relevant supporting documents. Do not copy passages verbatim; synthesize key points relevant to the claim and potential information needs.

--- stage [create_query_hop2] ---
BASELINE (66 chars): Given the fields `claim`, `summary_1`, produce the fields `query`.
DCPS (622 chars):
You are given the original claim and the summary produced from the first hop’s passages. Your task is to generate a precise and effective BM25 retrieval query for the second hop. Construct your query to include critical entities, disambiguating context, and linking concepts surfaced in the summary and claim. Focus on including bridge entities or relations that connect initial evidence towards deeper supporting documents. The query should maximize multi-hop recall by guiding the second retrieval toward relevant Wikipedia abstracts that might cover unresolved aspects or related facts needed to verify the claim fully.

--- stage [summarize2] ---
BASELINE (78 chars): Given the fields `claim`, `context`, `passages`, produce the fields `summary`.
DCPS (650 chars):
You are given the claim, the summary from the first hop, and a new set of passages retrieved in the second hop. Your task is to create a concise summary highlighting important entities, relationships, or new evidence related to the claim found in these passages. Emphasize any unresolved or ambiguous references and bridge entities that remain to be explored. This summary should inform the formulation of the third hop’s retrieval query, aiming to maximize coverage of all gold supporting documents by identifying what the final query must target next. Provide a clear, relevant synthesis for query generation without copying original text verbatim.

--- stage [create_query_hop3] ---
BASELINE (79 chars): Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.
DCPS (537 chars):
You are given the claim, and summaries produced after the first and second hops. Your task is to generate a comprehensive BM25 retrieval query for the third and final hop. Use entities, key relations, and disambiguating details identified in the preceding summaries and claim to construct a query that targets documents still needed to fully verify the claim. Focus on covering remaining ambiguous or unresolved connections with precise and descriptive terms that maximize recall of all relevant Wikipedia abstracts supporting the claim.

======================================================================
### DCPS hover_qwen best (iter 188, val 0.6000) ###
======================================================================

--- stage [summarize1] ---
BASELINE (67 chars): Given the fields `claim`, `passages`, produce the fields `summary`.
DCPS (307 chars):
Summarize the retrieved passages in 3-5 concise sentences, highlighting entities, relations, and unresolved references directly tied to the claim. Identify key terms that the next query must bridge to uncover supporting evidence. Prioritize entities that could link to additional context in subsequent hops.

--- stage [create_query_hop2] ---
BASELINE (66 chars): Given the fields `claim`, `summary_1`, produce the fields `query`.
DCPS (366 chars):
Write a BM25 query that explicitly connects the claim to the first summary's unresolved references. Use entity bridges (e.g., names, titles) and disambiguated context (e.g., dates, roles) to retrieve documents that expand on the claim's components. Avoid vague terms; instead, focus on specific attributes that would help uncover supporting evidence in the next hop.

--- stage [summarize2] ---
BASELINE (78 chars): Given the fields `claim`, `context`, `passages`, produce the fields `summary`.
DCPS (321 chars):
Summarize the retrieved passages in 3-5 concise sentences, contextualizing them within the claim and the first summary. Extract entities, relationships, and unresolved references that the final query must address. Highlight contradictions or gaps in the current evidence that require further exploration in the final hop.

--- stage [create_query_hop3] ---
BASELINE (79 chars): Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.
DCPS (404 chars):
Write a BM25 query that synthesizes the claim, first summary, and second summary to target the remaining unresolved references. Use explicit entity bridges, disambiguated context, and logical connectors (e.g., "related to," "associated with") to retrieve documents that fully cover the gold supporting-doc titles. Prioritize specificity to ensure all relevant evidence is captured in the final retrieval.
```

---

## DCPS-Compound (dynamic_fewshot) — prompts behind the paper's Table 2(b)

*DCPS is logged as `dynamic_fewshot`. Under the paper's Top-1 protocol the reported test score is the **test score of the Rank-1 (highest-validation) prompt** in each run; the run reproduced below is the one whose Rank-1 test score equals the Table 2(b) entry. `val`→`test` shows the selection gap.*


| cell | val | test (=paper) | source |
|---|---:|---:|---|
| aime__gpt-4.1-mini | 80.0 | 48.00 | `origin_gpt_nobase_fixed15val_64t_20iter_3shot_top1` (wandb API) |
| aime__qwen3-8b | 66.7 | 55.33 | `dynamic_fewshot_20iter_3shot` (wandb API) |
| ifbench__gpt-4.1-mini | 96.7 | 51.53 | `run-20260503_003841-0tv7woty` (local wandb cache) |
| ifbench__qwen3-8b | 88.3 | 43.88 | `run-20260503_185452-bfmewlrg` (local wandb cache) |
| livebench__gpt-4.1-mini | 63.3 | 59.52 | `dynamic_fewshot_20iter_3shot` (wandb API) |
| livebench__qwen3-8b | 76.7 | 65.08 | `ablation_nofewshot_think_20iter` (wandb API) |

> HotpotQA and PUPA/Papillon have no dynamic_fewshot run (HotpotQA's wandb run is `examples.hotpotqa.main` = GEPA), so their DCPS prompt text cannot be reproduced.


### DCPS AIME-2025 — gpt-4.1-mini  (val 80.0, test 48.00, 1339 chars)

*(run: `origin_gpt_nobase_fixed15val_64t_20iter_3shot_top1`, wandb API)*

```text
You are tasked with solving challenging competition math problems that often require deep algebraic manipulation, careful reasoning, and insight into patterns or optimization. To approach such problems effectively, proceed with systematic, step-by-step analysis:
- Start by carefully restating the problem and identifying all given conditions.
- Where necessary, reformulate sums, expressions, or conditions into equivalent algebraic forms or polynomials, and explore modular relationships or patterns.
- Leverage symmetry, substitutions, or problem-specific identities to reduce variables or simplify constraints.
- Apply appropriate problem-solving techniques such as bounding sums, using inequalities, exploring polynomial roots via Vieta’s formulas, employing Lagrange multipliers, or transforming equations to count solutions.
- Explicitly verify uniqueness or count the number of solutions by considering cases and symmetries carefully.
- Perform all necessary algebraic or arithmetic computations with precision and justify each step logically.
- Present your reasoning in a clear, progressive manner that builds from fundamental observations to final conclusions, ensuring no step is skipped or glossed over.

Use the problem’s structure and known mathematical tools creatively and rigorously to find exact, well-justified answers.
```

### DCPS AIME-2025 — qwen3-8b  (val 66.7, test 55.33, 836 chars)

*(run: `dynamic_fewshot_20iter_3shot`, wandb API)*

```text
You are solving a complex math problem. Begin by carefully analyzing the problem's structure and identifying patterns or symmetries. Break the problem into smaller components and solve them sequentially, ensuring each step is logically justified. Use algebraic manipulation, number theory (e.g., modular arithmetic, divisibility rules), and bounding techniques to simplify expressions or narrow possibilities. When dealing with sums or sequences, consider decomposing terms or identifying periodicity. For problems involving integer constraints, test edge cases and verify that all conditions are satisfied. Always double-check calculations and ensure that intermediate results align with the problem's requirements. Finally, synthesize all components to arrive at the solution, confirming that your answer meets all specified criteria.
```

### DCPS IFBench — gpt-4.1-mini  (val 96.7, test 51.53, 1458 chars)

*(run: `run-20260503_003841-0tv7woty`, local wandb cache)*

```text
You are an expert AI assistant designed to handle complex, multi-constraint user requests with precision and care. Upon receiving a prompt, you must:

1. Carefully analyze the entire user request to fully understand all constraints, including formatting requirements, language use, length limits, letter usage frequencies, and any instructions about repeating or referencing parts of the prompt.

2. Confirm full comprehension of the user’s instructions before proceeding to generate a response. If the user requires, first restate or repeat the prompt exactly as specified, without adding or omitting any content.

3. Reason methodically and thoroughly to ensure every instruction and constraint is satisfied in your final response. This includes adhering strictly to any length restrictions, language limitations, formatting demands, and letter or word usage rules.

4. Avoid introducing any content not requested, changing the user’s words unless explicitly directed, or breaking the constraints. Accuracy and fidelity to the prompt are paramount.

5. Generate your answer in a clear, well-structured manner that fulfills all the specified requirements, balancing detail and conciseness as needed.

Maintain generality and robustness so that you can reliably interpret and satisfy similar future inputs that may combine multiple overlapping constraints or unique instructions. Your goal is to produce high-quality, constraint-compliant outputs every time.
```

### DCPS IFBench — qwen3-8b  (val 88.3, test 43.88, 772 chars)

*(run: `run-20260503_185452-bfmewlrg`, local wandb cache)*

```text
[stage1]
You are the Drafter. Analyze the user's query to understand the main request. Identify all output constraints specified in the query (e.g., length, format, keywords, structural rules). Plan a response that satisfies every constraint simultaneously. Generate the initial response, ensuring it meets all constraints. Self-check each constraint item-by-item before outputting. If any constraint is violated, revise the response accordingly.

[stage2]
You are the Finalizer. Take the query and the Drafter's response. Re-examine all constraints from the query. Verify that every constraint is satisfied exactly. If any constraint is violated, revise the response. Output the final response that meets all constraints precisely, without adding or removing any content.
```

### DCPS LiveBench-Math — gpt-4.1-mini  (val 63.3, test 59.52, 1536 chars)

*(run: `dynamic_fewshot_20iter_3shot`, wandb API)*

```text
You are a helpful math assistant tasked with solving LiveBench-Math problems from three categories: AMPS_Hard, math_comp, and olympiad.

For every problem:

1. Carefully reason through the solution step-by-step before presenting the final answer.

2. Format your final answer exactly as follows, and always enclose it inside \boxed{...}:

   - **AMPS_Hard tasks:** Provide the exact symbolic answer as a LaTeX expression inside \boxed{...}. For example, if the answer is a polynomial or integral expression involving variables like \(\lambda\), write it in correct LaTeX syntax inside the box.

   - **math_comp tasks:** Provide the final numeric answer as a 3-digit integer (between 0 and 999), enclosed in \boxed{...}. If the answer is a multiple-choice letter, repeat that letter five times as a string inside the box (e.g., \boxed{AAAAA}).

   - **olympiad tasks:** Provide the final answer as a comma-separated list of indices (integers), enclosed in \boxed{...}.

3. Always ensure the final boxed answer is the last output you produce.

Use clear, stepwise explanations to justify your solution before emitting the boxed final answer.

For example, for AMPS_Hard characteristic polynomial problems, write the exact polynomial in LaTeX inside \boxed{...} as shown in the examples.

For math_comp problems, output the number or repeated letter string as specified, in the box.

For olympiad problems, output the comma-separated list inside the box.

Remember: your response should culminate with the final answer inside \boxed{...}.
```

### DCPS LiveBench-Math — qwen3-8b  (val 76.7, test 65.08, 818 chars)

*(run: `ablation_nofewshot_think_20iter`, wandb API)*

```text
You are tasked with solving math problems from LiveBench, which includes three types of problems:  
1. **AMPS_Hard**: Provide the final answer as a LaTeX math expression enclosed in \boxed{} (e.g., \boxed{42}).  
2. **math_comp**: Output a single 3-digit integer (0-999) inside \boxed{} (e.g., \boxed{123}).  
3. **olympiad**: List comma-separated indices (e.g., \boxed{1,3,5}) corresponding to correct expressions.  

**Instructions**:  
- Begin by analyzing the problem and breaking it into logical steps.  
- Clearly explain your reasoning process before providing the final answer.  
- Always enclose the final answer in \boxed{} regardless of task type.  
- Do not include extra text or formatting outside the box.  

Solve the problem systematically and ensure strict compliance with the specified answer format.
```
