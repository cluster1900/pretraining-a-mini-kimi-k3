"""Map a multiple-choice answer to a choice index without loading the model."""


def resolve_choice_index(answer, choices, line_no):
    if isinstance(answer, str):
        if answer in choices:
            return choices.index(answer)
        letter = answer.strip().upper()
        if len(letter) == 1 and "A" <= letter <= "Z":
            index = ord(letter) - ord("A")
            if index < len(choices):
                return index
        raise ValueError(f"Line {line_no} answer is not one of the choices or a letter label")
    if isinstance(answer, int) and 0 <= answer < len(choices):
        return answer
    raise ValueError(f"Line {line_no} answer index is outside the choices")
