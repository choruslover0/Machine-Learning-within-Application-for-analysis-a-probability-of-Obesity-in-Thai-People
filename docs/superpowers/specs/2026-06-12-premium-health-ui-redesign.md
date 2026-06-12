# O-Beast Premium Health UI Redesign

## Goal

Redesign O-Beast as a calm, premium, user-facing health research app. Users should understand the app purpose, complete the obesity-risk check without form fatigue, read results quickly, and learn how the system works without excessive technical language.

The redesign changes presentation and interaction only. Existing prediction, training, risk-tier, and advice logic remain unchanged.

## Selected Direction

**Premium Health** combines:

- Soft structural visual design
- Deep green, sage, warm white, and small peach accents
- Premium geometric typography
- Airy layouts with clear visual hierarchy
- Soft depth instead of strong gradients or harsh shadows
- Calm motion that supports understanding

The interface should feel trustworthy and modern without appearing clinical, intimidating, or developer-focused.

## Audience

- Students and general users with no machine-learning knowledge
- Teachers and professors reviewing the research project
- Researchers reviewing model methods and evaluation details

Primary pages prioritize everyday users. Technical evidence remains available on the Methods page.

## Information Architecture

### Home

Purpose: introduce O-Beast and move users into prediction.

- Clear headline: understand habits and obesity-risk probability
- One primary action: begin risk check
- One secondary action: understand how O-Beast works
- Three concise facts: answer count, compared models, clear result
- Simplified algorithm-learning section
- Wellness advice preview
- Producer profiles and contact links

### Predictor

Purpose: collect answers with minimal form fatigue.

- Divide existing inputs into guided groups:
  1. Body profile
  2. Daily routine
  3. Food habits and family history
  4. Review and submit
- Show visible progress
- Use plain-language labels and short supporting descriptions
- Preserve current input names, validation, and form submission endpoint
- Provide clear previous/continue controls
- Keep one final submit action

JavaScript controls step visibility only. All inputs remain in one form so backend behavior stays unchanged.

### Result

Purpose: explain probability and next actions clearly.

Order:

1. Estimated probability
2. Five-level risk band
3. Short explanation of what result means
4. Three most useful wellness actions
5. Full personalized advice
6. Model transparency and comparison details

Technical metrics must not dominate initial result viewport.

### Methods

Purpose: provide research transparency.

- Explain model comparison in plain language first
- Show simplified algorithm visuals
- Show current model, validation, balancing, and dataset limitations
- Keep evaluation bars and metric table
- Use disclosure-style grouping to reduce visual overload

### Advice

Purpose: explain transparent wellness-guidance logic.

- Show how user answers become recommendations
- Group advice by movement, food, sleep, and screen habits
- Show trusted sources
- Clearly state advice is educational, not diagnosis

## Visual System

### Color

- Ink: `#14241B`
- Deep green: `#153C29`
- Action green: `#347849`
- Sage surface: `#E9F0E8`
- Warm white: `#F7FAF6`
- Peach accent: `#EAA377`
- Muted text: `#66746C`

Risk colors remain distinguishable and accessible. Risk meaning must never depend on color alone.

### Typography

- Display and headings: Manrope
- Body and controls: DM Sans
- Use responsive, restrained type sizes
- Avoid oversized text inside compact components

### Components

- Floating compact navigation
- Nested-shell cards: soft outer enclosure plus clean inner surface
- Pill primary actions with contained circular arrow icon
- Consistent form fields with strong labels
- Stable probability ring and risk-band badge
- Light algorithm diagrams with small arrows and short labels

### Motion

- Page and section entry: gentle fade and vertical transform
- Buttons: subtle press and contained icon movement
- Predictor steps: opacity and transform transitions
- Respect `prefers-reduced-motion`
- Animate transform and opacity only

## Higgsfield Asset

Generate one abstract premium health-pattern image using Higgsfield GPT Image 2.

Asset purpose:

- Support home-page learning section
- Visualize body and lifestyle signals becoming an understandable pattern
- Avoid medical imagery, body-shaming, text, UI screenshots, or diagnostic claims
- Match deep green, sage, warm white, and peach palette

Generated asset is decorative and explanatory. Core UI remains usable if asset fails to load.

## User-Facing Writing Rules

- Speak directly to users
- Use simple health and machine-learning language
- Never include development commands, implementation notes, private instructions, or conversation history
- Avoid universal claims about model superiority
- State model is selected for current training data
- State prediction and advice are educational, not medical diagnosis

## Accessibility

- Keyboard-accessible navigation and predictor steps
- Visible focus indicators
- Semantic headings, labels, buttons, and links
- Minimum comfortable touch targets
- Sufficient contrast
- Meaningful image alternative text
- Reduced-motion support
- Mobile layout becomes single column without overlap or horizontal clipping

## Technical Approach

- Keep FastAPI route structure and Python HTML generation
- Refactor shared CSS tokens and common UI components inside `src/obesity_ml/app.py`
- Add generated visual under `src/obesity_ml/static/`
- Add small JavaScript controller for predictor steps
- Preserve endpoints, field names, prediction request, chatbot, and ML behavior
- Extend regression tests for page text, form fields, and critical controls

## Verification

- Python compile check
- Existing unit and regression tests
- HTTP `200` smoke checks for Home, Predictor, Advice, and Methods
- Prediction form submission smoke check
- Desktop and mobile browser screenshots
- Check no overlap, clipping, blank assets, or unusable controls
- Check keyboard navigation and reduced-motion behavior
- `git diff --check`

## Out of Scope

- Changing ML algorithms, training data, risk thresholds, or advice logic
- Adding authentication or user accounts
- Replacing FastAPI architecture
- Making medical accuracy or diagnosis claims

