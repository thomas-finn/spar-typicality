# Data for typicality experiments

## Raw data

In `./raw` we store datasets borrowed from pyschology literature studying typicality through category-item pairs.

 - Rosch, 1975

## Processed data

For each dataset with category-item pairs we construct a suite of prompts that we store in `./processed`.
Each type of prompt is summarised in the table below:

| Prompt set | Template |
|:-|:-|
| `{dataset}_label`          | `{category}: {item}.` |
| `{dataset}_neutral`        | `Word: {item}.` |
| `{dataset}_example`        | `A {item} is a {category}.` |
| `{dataset}_neg`            | `A {item} is not a {category}.` |
| `{dataset}_paris_true`     | `A {item} is a {category} and Paris is in France.` |
| `{dataset}_paris_false`    | `A {item} is a {category} and Paris is in China.` |
| `{dataset}_beijing_true`   | `A {item} is a {category} and Beijing is in China.` |
| `{dataset}_beijing_false`  | `A {item} is a {category} and Beijing is in France.` |
| `{dataset}_addition_true`  | `A {item} is a {category} and 11+10=21.` |
| `{dataset}_addition_false` | `A {item} is a {category} and 11+10=25.` |
| `{dataset}_story`          | `Once upon a time there was a girl named Lila. In her house there was a magical room that contained {category} and when she entered she saw a {item}.` |

**Note**: Each template is altered when necessary to maintain grammatical sense. For example, `A hat is a clothing.` would be altered to `A hat is a piece of clothing.`.

Each prompt set `name` is contained in  `name.csv` with the fields
 - `prompt`: String containing the prompt 
 - `item`: The item inserted into the prompt
 - `category`: The category inserted into the prompt (`None` if the prompt does not contain a category)
 - `typicality_rating_raw`: The unnormalised typicality rating of the item in a the given category from the original dataset
 - `typicality_rating_normalised`: The z-score of the typicality rating of the item in the given category
 - `is_member`: Boolean flag for whether the item is a member of the category (`None` if no category in the prompt)

For each prompt set there will be the category-items pairs from the raw dataset.
Additionally there will be 100 examples of random item-category pairs that will have:
 - `None` for `typicality_rating_raw` and `typicality_rating_normalised`
 - `0` for `is_member`
The idea of these 100 examples is to see how the prompts are represented when nonsense is given where the item would not be a member of the category.
