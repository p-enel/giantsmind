from typing import List

from collection_operations import (
    act_add_results_to_paper_context,
    act_add_to_collection,
    act_apply_selection,
    act_load_collection,
    act_select_papers,
    act_set_results_as_working_papers,
    get_collections,
    save_collection,
    act_create_new_collection,
    get_all_papers_collection,
)
from context import Context, context_updater
from state import states
from helpers import Options, state_handler, list2optargs
import interact_operations as interact_ops
from search_operations import (
    act_refine_search,
    act_search_with_content,
    act_search_with_metadata,
)


def state_func_SEARCH_PAPERS(context: Context) -> Context:

    if not context.search_scope:
        context.search_scope = context.papers

    options = list2optargs(
        [
            (
                "Search with Metadata",
                lambda ctx: act_search_with_metadata(context=ctx),
            ),
            (
                "Search with Content",
                lambda ctx: act_search_with_content(context=ctx),
            ),
            ("Load Collection", context_updater({"current_state": states.LOAD_COLLECTION})),
            (
                "Interact with Paper",
                context_updater(
                    {"current_state": states.INTERACT_WITH_PAPERS, "state_context": {}},
                ),
            ),
            ("Create a New Collection", act_create_new_collection),
        ]
    ) + [("q", "Quit", context_updater({"current_state": states.EXIT}))]
    return state_handler(context, Options(options))


def state_func_LOAD_COLLECTION(context: Context) -> Context:

    collections: List[str] = get_collections()[1]
    options = list2optargs(
        [
            (
                f"Collection '{collection}'",
                lambda ctx: act_load_collection(ctx, collection),
            )
            for collection in collections
        ]
    ) + [
        ("c", "Cancel", context_updater({"current_state": states.SEARCH_PAPERS})),
    ]
    return state_handler(context, Options(options))


def state_func_AFTER_SEARCH(context: Context) -> Context:
    options = list2optargs(
        [
            ("Refine Search", act_refine_search),
            (
                "New Search",
                context_updater(
                    {
                        "current_state": states.SEARCH_PAPERS,
                        "search_scope": get_all_papers_collection(),
                    }
                ),
            ),
            ("Set Results as Context", act_set_results_as_working_papers),
            (
                "Select Papers",
                context_updater(
                    {
                        "current_state": states.SELECT_PAPERS,
                        "results": context.results,
                        "selected_papers": None,
                    }
                ),
            ),
            ("Add Results to Context", act_add_results_to_paper_context),
            ("Create a New Collection", act_create_new_collection),
            ("Add to Collection", act_add_to_collection),
            (
                "Interact with Papers",
                context_updater(
                    {
                        "current_state": states.INTERACT_WITH_PAPERS,
                        "results": None,
                    }
                ),
            ),
            (
                "Search Again",
                context_updater(
                    {
                        "current_state": states.SEARCH_PAPERS,
                        "state_context": {},
                    }
                ),
            ),
        ]
    ) + [("q", "Quit", context_updater({"current_state": states.EXIT}))]
    return state_handler(context, Options(options))


def state_func_SELECT_PAPERS(context: Context) -> Context:
    options = list2optargs(
        [
            ("Select Papers to Keep", lambda ctx: act_select_papers(ctx, keep=True)),
            ("Select Papers to Discard", lambda ctx: act_select_papers(ctx, keep=False)),
            ("Apply Selection", act_apply_selection),
        ]
    ) + [("c", "Cancel", context_updater({"current_state": context.previous_state}))]
    return state_handler(context, Options(options))


def state_func_INTERACT_WITH_PAPERS(context: Context) -> Context:

    if not context.papers:
        print("No papers to interact with.")
        new_context = context.copy()
        new_context.current_state = new_context.previous_state
        return new_context

    options = list2optargs(
        [
            ("Ask Question", interact_ops.act_ask_question),
            ("Set Paper Context", context_updater({"current_state": states.SEARCH_PAPERS})),
            (
                "Select Papers",
                context_updater(
                    {
                        "current_state": states.SELECT_PAPERS,
                        "selected_papers": None,
                    },
                ),
            ),
        ]
    ) + [("q", "Quit", context_updater({"current_state": states.EXIT}))]

    return state_handler(context, Options(options))


def state_func_EXIT(context: Context) -> Context:
    # input_msg = "Do you want to save the current papers in a collection before exiting? (y/[n])\n"
    # choice = input(input_msg)

    # if choice.lower() == "y":
    #     to_save_mapping = {
    #         states.SELECT_PAPERS: context.state_context.get("papers_selected"),
    #         states.AFTER_SEARCH: context.state_context.get("results"),
    #     }
    #     to_save = to_save_mapping.get(context.current_state, context.papers)
    #     if not to_save:
    #         print("No papers to save.")
    #         return context
    #     save_collection(to_save)
    print("Exiting the program...")
    return context
