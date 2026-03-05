from typing import Optional

# Seed templates covering the most common TWiki relation types
_DEFAULT_TEMPLATES = {
    "composer": "The composer of {subject} is",
    "ceo": "The CEO of {subject} is",
    "capital": "The capital of {subject} is",
    "country": "The country of {subject} is",
    "director": "The director of {subject} is",
    "author": "The author of {subject} is",
    "publisher": "The publisher of {subject} is",
    "manufacturer": "The manufacturer of {subject} is",
    "owner": "The owner of {subject} is",
    "headquarters": "The headquarters of {subject} is located in",
    "founded": "{subject} was founded in",
    "chairperson": "The chairperson of {subject} is",
    "parent_organization": "The parent organization of {subject} is",
    "member_of": "{subject} is a member of",
    "spouse": "The spouse of {subject} is",
    "employer": "The employer of {subject} is",
    "place_of_birth": "{subject} was born in",
    "place_of_death": "{subject} died in",
    "educated_at": "{subject} was educated at",
    "occupation": "The occupation of {subject} is",
    "nationality": "The nationality of {subject} is",
    "genre": "The genre of {subject} is",
    "language": "The language of {subject} is",
    "currency": "The currency of {subject} is",
    "religion": "The religion of {subject} is",
    "population": "The population of {subject} is",
    "area": "The area of {subject} is",
    "official_language": "The official language of {subject} is",
    "head_of_government": "The head of government of {subject} is",
    "head_of_state": "The head of state of {subject} is",
    "anthem": "The anthem of {subject} is",
    "flag": "The flag of {subject} is",
    "coat_of_arms": "The coat of arms of {subject} is",
    "motto": "The motto of {subject} is",
    "inception": "{subject} was established in",
    "dissolved": "{subject} was dissolved in",
    "follows": "{subject} follows",
    "followed_by": "{subject} is followed by",
    "part_of": "{subject} is part of",
    "has_part": "{subject} has part",
    "instance_of": "{subject} is an instance of",
    "subclass_of": "{subject} is a subclass of",
    "located_in": "{subject} is located in",
    "located_on": "{subject} is located on",
    "adjacent_to": "{subject} is adjacent to",
    "shares_border_with": "{subject} shares a border with",
    "cast_member": "A cast member of {subject} is",
    "producer": "The producer of {subject} is",
    "screenwriter": "The screenwriter of {subject} is",
    "distributor": "The distributor of {subject} is",
    "platform": "The platform of {subject} is",
    "operating_system": "The operating system of {subject} is",
    "developer": "The developer of {subject} is",
    "programming_language": "{subject} is written in",
    "license": "The license of {subject} is",
    "award_received": "{subject} received the award",
    "nominated_for": "{subject} was nominated for",
    "field_of_work": "The field of work of {subject} is",
    "work_location": "The work location of {subject} is",
    "position_held": "The position held by {subject} is",
    "military_rank": "The military rank of {subject} is",
    "conflict": "{subject} participated in the conflict",
    "participant": "A participant in {subject} is",
    "point_in_time": "{subject} occurred at",
    "start_time": "{subject} started at",
    "end_time": "{subject} ended at",
    "duration": "The duration of {subject} is",
    "series": "{subject} is part of the series",
    "season": "{subject} is in season",
    "episode": "{subject} is episode",
    "number_of_episodes": "The number of episodes of {subject} is",
    "network": "{subject} airs on",
    "record_label": "The record label of {subject} is",
    "album": "The album containing {subject} is",
    "tracklist": "A track on {subject} is",
    "lyrics_by": "The lyrics of {subject} were written by",
    "performer": "The performer of {subject} is",
    "sport": "The sport of {subject} is",
    "league": "{subject} plays in the league",
    "team": "{subject} plays for the team",
    "coach": "The coach of {subject} is",
    "manager": "The manager of {subject} is",
    "home_venue": "The home venue of {subject} is",
    "country_of_origin": "The country of origin of {subject} is",
    "original_language": "The original language of {subject} is",
    "publication_date": "{subject} was published on",
    "edition": "The edition of {subject} is",
    "isbn": "The ISBN of {subject} is",
    "main_subject": "The main subject of {subject} is",
    "depicts": "{subject} depicts",
    "material_used": "The material used in {subject} is",
    "color": "The color of {subject} is",
    "shape": "The shape of {subject} is",
    "weight": "The weight of {subject} is",
    "height": "The height of {subject} is",
    "width": "The width of {subject} is",
    "depth": "The depth of {subject} is",
    "diameter": "The diameter of {subject} is",
    "length": "The length of {subject} is",
}


class Verbalizer:
    def __init__(self):
        self._templates: dict[str, str] = dict(_DEFAULT_TEMPLATES)

    def verbalize(self, subject: str, relation: str) -> Optional[str]:
        """Return cloze prompt with object masked. None if no template exists."""
        template = self._templates.get(relation)
        if template is None:
            return None
        return template.format(subject=subject)

    def register(self, relation: str, template: str) -> None:
        """Register a new relation->template mapping. Template must contain {subject}."""
        if "{subject}" not in template:
            raise ValueError(f"Template must contain {{subject}} placeholder: {template!r}")
        self._templates[relation] = template

    def coverage(self) -> float:
        """Fraction of registered templates (always 1.0 for registered set; meaningful when checked against external relation list)."""
        return len(self._templates) / max(len(self._templates), 1)

    def coverage_over(self, relations: list[str]) -> float:
        """Fraction of a given relation list that has a registered template."""
        if not relations:
            return 0.0
        covered = sum(1 for r in relations if r in self._templates)
        return covered / len(relations)

    def known_relations(self) -> list[str]:
        return list(self._templates.keys())