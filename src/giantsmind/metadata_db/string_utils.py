import Levenshtein


def levenshtein(s1: str, s2: str) -> int:
    return Levenshtein.distance(s1.lower(), s2.lower())


def author_name_distance(db_name: str, query_name: str) -> int:
    db_name = db_name.lower()
    query_name = query_name.lower()

    db_parts = db_name.split()
    query_parts = query_name.split()

    if len(query_parts) == 1:
        if len(db_parts) > 1:
            return levenshtein(db_parts[1], query_name)
        return levenshtein(db_name, query_name)

    elif len(query_parts) == 2:
        if len(db_parts) >= 2:
            normal_order = levenshtein(db_parts[0], query_parts[0]) + levenshtein(
                db_parts[-1], query_parts[1]
            )
            swapped_order = levenshtein(db_parts[0], query_parts[1]) + levenshtein(
                db_parts[-1], query_parts[0]
            )
            return min(normal_order, swapped_order)
        return min(
            levenshtein(db_name, query_name),
            levenshtein(db_name, f"{query_parts[1]} {query_parts[0]}"),
        )

    return levenshtein(db_name, query_name)
