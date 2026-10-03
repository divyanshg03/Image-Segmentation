# Sample images

Small images used by the demo notebook, the tests and the benchmark.

| file | size | what it is for |
|---|---|---|
| `football_dribble.jpeg` | 735x471 | Multi-object scene. Clean prompts (points, box) on a single player; a click on the pitch segments the pitch. Used by the tests and the benchmark. |
| `trophy_celebration.jpeg` | 736x414 | Crowded photo; a box around the player lifting the trophy. |
| `starry_night.jpg` | 736x414 | Painting: soft, textured boundaries (the cypress tree is a good box target). |
| `wolf_of_wall_street_poster.jpg` | 562x321 | Dense illustration with text. Segment-everything mode fragments it; a prompt on the central figure works but SAM is least confident here. |

## Provenance and licensing

These images were collected from the web for experimentation and **their copyright
belongs to their respective owners**. They are not covered by this repository's
Apache-2.0 license. They are kept here only as small educational test inputs; replace
them with your own photos or openly licensed images (e.g. COCO, Unsplash, Wikimedia
Commons) before redistributing the project.

The tests refer to these files by name and by pixel coordinates, so if you swap one
out, update the coordinates in `tests/test_segmenter.py` and the notebook as well.
