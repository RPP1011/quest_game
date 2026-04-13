from pathlib import Path
import textwrap
import pytest
from app.craft.library import CraftLibrary
from app.craft.schemas import Structure, Tool


@pytest.fixture
def tiny_library_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "structures").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)
    (root / "examples").mkdir(parents=True)
    (root / "styles").mkdir(parents=True)

    (root / "structures" / "tiny.yaml").write_text(textwrap.dedent("""
        id: tiny
        name: Tiny
        description: A minimal test structure.
        scales: [scene]
        phases:
          - name: open
            position: 0
            tension_target: 0.2
            expected_beats: [scene_sequel]
            description: Opening.
          - name: close
            position: 1
            tension_target: 0.8
            description: Close.
        tension_curve:
          - [0.0, 0.1]
          - [1.0, 0.8]
    """))

    (root / "tools" / "scene_sequel.yaml").write_text(textwrap.dedent("""
        id: scene_sequel
        name: Scene and Sequel
        category: pacing
        description: Scene then reaction.
        preconditions: [A just-resolved action]
        signals: [Reader needs to metabolize what just happened]
        anti_patterns: [Skipping the sequel entirely]
        example_ids: [ex_sq1]
    """))

    (root / "examples" / "beats.yaml").write_text(textwrap.dedent("""
        examples:
          - id: ex_sq1
            tool_ids: [scene_sequel]
            source: original
            scale: scene
            snippet: |
              The door shut behind him. For a long moment she simply stood,
              her hand still raised as if to call him back. Then she lowered it,
              and began, slowly, to clear the cups.
            annotation: Scene ends on action; sequel is the pause + the small task.
    """))

    (root / "styles" / "terse.yaml").write_text(textwrap.dedent("""
        id: terse
        name: Terse
        description: Dry, declarative, low interiority.
        sentence_variance: low
        concrete_abstract_ratio: 0.8
        interiority_depth: surface
        pov_discipline: strict
        diction_register: plain
        voice_samples:
          - |
            He walked. The fire was out. He counted the bullets.
    """))
    return root


def test_load_indexes_everything(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    assert isinstance(lib.structure("tiny"), Structure)
    assert isinstance(lib.tool("scene_sequel"), Tool)
    assert lib.example("ex_sq1").scale == "scene"
    assert lib.style("terse").sentence_variance == "low"


def test_getters_raise_for_missing(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    with pytest.raises(KeyError):
        lib.structure("nope")
    with pytest.raises(KeyError):
        lib.tool("nope")


def test_filter_by_scale(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    scene_structs = lib.structures(scale="scene")
    assert [s.id for s in scene_structs] == ["tiny"]
    assert lib.structures(scale="campaign") == []


def test_filter_tools_by_category(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    pacing = lib.tools(category="pacing")
    assert [t.id for t in pacing] == ["scene_sequel"]
    assert lib.tools(category="reversal") == []


def test_examples_for_tool(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    examples = lib.examples_for_tool("scene_sequel")
    assert [e.id for e in examples] == ["ex_sq1"]
    assert lib.examples_for_tool("unknown") == []


def test_load_rejects_duplicate_ids(tiny_library_root: Path):
    (tiny_library_root / "tools" / "dup.yaml").write_text(
        "id: scene_sequel\nname: Dup\ncategory: pacing\ndescription: x\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        CraftLibrary(tiny_library_root)


def test_real_structures_load():
    from app.craft.library import CraftLibrary
    lib = CraftLibrary(Path(__file__).parent.parent.parent / "app" / "craft" / "data")
    ids = {s.id for s in lib.structures()}
    assert {"three_act", "five_act_freytag", "kishotenketsu"} <= ids
    three = lib.structure("three_act")
    assert len(three.phases) == 5


def test_real_tools_load():
    from app.craft.library import CraftLibrary
    lib = CraftLibrary(Path(__file__).parent.parent.parent / "app" / "craft" / "data")
    ids = {t.id for t in lib.tools()}
    expected = {"chekhovs_gun", "try_fail_cycle", "reversal",
                "scene_sequel", "midpoint_shift", "false_victory"}
    assert expected <= ids
    # Categories distributed
    cats = {t.category for t in lib.tools()}
    assert {"foreshadowing", "pacing", "reversal", "structural", "tension"} <= cats


def test_real_examples_load_and_link_tools():
    from app.craft.library import CraftLibrary
    lib = CraftLibrary(Path(__file__).parent.parent.parent / "app" / "craft" / "data")
    assert len(lib.all_examples()) >= 10
    chek = lib.examples_for_tool("chekhovs_gun")
    assert len(chek) >= 2
    # Every example references only real tool ids
    tool_ids = {t.id for t in lib.tools()}
    for ex in lib.all_examples():
        for tid in ex.tool_ids:
            assert tid in tool_ids, f"example {ex.id} references unknown tool {tid}"
