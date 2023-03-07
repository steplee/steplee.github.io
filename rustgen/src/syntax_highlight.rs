use std::path::PathBuf;
use tree_sitter::Parser;
use tree_sitter::TreeCursor;
use std::collections::BTreeMap;

/*

   I use treesitter to parse the code.
   Then I look for terminal nodes (and treat some non-terminal subtrees as terminals) and
   create <span>s for them, with appropriate class.

   Each language has different keywords and such, but they share the same CSS colorscheme and classes.

*/

// The kinds are sort of arbitrary. I'll need a list of keywords for each lang, as the output is
// fine-grained.
fn dbg_get_kinds(lang: &tree_sitter::Language) -> BTreeMap<String, u16> {
    let mut map = BTreeMap::new();
    for i in 0..=lang.node_kind_count() {
        if let Some(name) = lang.node_kind_for_id(i as u16) {
            map.insert(String::from(name), i as u16);
        }
    }
    return map;
}

//
// NOTE: Count 'string_literal' as a terminal node (even though it has children)
//
fn visit(c:&mut TreeCursor, src: &[u8], f: &mut dyn FnMut(&tree_sitter::Node) -> ()) {
    let node = c.node();
    let s = node.utf8_text(src).unwrap_or("").to_string();
    // println!("s: {} {} ({}/{}) {}", node.is_named() as u8, node.child_count(), node.kind(),node.kind_id(), s);

    if node.child_count() == 0 ||
       node.kind() == "preproc_def" ||
       node.kind() == "string_literal" ||
       node.kind() == "char_literal" {
        println!("s: {} {} ({}/{}) {}", node.is_named() as u8, node.child_count(), node.kind(),node.kind_id(), s);
        f(&node);
    } else if c.goto_first_child() {
        visit(c,src, f);
    }

    while c.goto_next_sibling() {
        visit(c,src, f);
    }

    c.goto_parent();
}

pub struct SyntaxHighlighter {

    pub shared_stylesheet: String,
    classes_per_lang: Vec<(Vec<String>,BTreeMap<u16,String>)>,

}

impl SyntaxHighlighter {


    pub fn new() -> SyntaxHighlighter {

        let shared_stylesheet = String::from(r#"
.c_kw {
    color: #aca;
    font-weight: bold;
}
.c_op {
    color: #99f;
}
.c_vr {
    color: #f99;
}
.c_ty {
    color: #99e;
}
.c_st {
    color: olive;
}
.c_fi {
    color: #e88;
}
.c_nl {
    color: orange;
}
.c_co {
    color: #77a;
}
.c_pp {
    color: #97a;
}
.c_mo {
    color: #a7e;
}
.code {
    margin: 10px;
    #background-color: #222;
    background-color: rgba(10, 4, 90, 0.1);
    #filter: brightness(170%);
    #backdrop-filter: brightness(170%);
}
"#);

        let mut classes_per_lang = vec![];

        // C++
        {
            let lang = tree_sitter_cpp::language();
            let kind_map = dbg_get_kinds(&lang);
            for (k,v) in kind_map.iter() {
                println!(" - {} -> {}", v,k);
            }
            let mut lang_classes: BTreeMap<u16,String> = BTreeMap::new();
            // lang_classes.insert(266 as u16, String::from("fi"));
            let keywords = vec!["if", "else", "break", "for", "class", "struct", "try", "catch", "goto", "const", "register", "auto", "return", "new", "delete", "static", "template", "constexpr", "using", "namespace", "switch", "case"];
            let types = vec!["primitive_type", "type_identifier"];
            let variables = vec!["identifier"];
            let operators = vec!["+", "-", "+=", "++", "--", "-=", "*", "/", "*=", "/=", "@", "^", "!", "!=", "^=", "=", "==", "%", "%=", ","];
            let strings = vec!["string_literal", "char_literal"];
            let literals = vec!["number_literal"];
            let field_ids = vec!["field_identifier"];
            let comments = vec!["comment"];
            let preproc = vec!["preproc_def"];
            let modules = vec!["namespace_identifier"];

            let pairs = [
                (keywords,"c_kw"),
                (types,"c_ty"),
                (variables,"c_vr"),
                (operators,"c_op"),
                (strings,"c_st"),
                (literals,"c_nl"),
                (field_ids,"c_fi"),
                (comments,"c_co"),
                (modules,"c_mo"),
                (preproc,"c_pp"),
            ];
            for (toks,code) in pairs.iter() {
                for &kw in toks.iter() {
                    if let Some(id) = kind_map.get(kw) {
                        lang_classes.insert(*id, String::from(*code));
                    }
                }
            }
            classes_per_lang.push((vec![String::from("CPP"),String::from("C++")], lang_classes));
        }

        // C
        {
            let lang = tree_sitter_c::language();
            let kind_map = dbg_get_kinds(&lang);
            for (k,v) in kind_map.iter() {
                println!(" - {} -> {}", v,k);
            }
            let mut lang_classes: BTreeMap<u16,String> = BTreeMap::new();
            // lang_classes.insert(266 as u16, String::from("fi"));
            let keywords = vec!["if", "else", "break", "for", "struct", "goto", "const", "register", "auto", "return", "static", "constexpr", "switch", "case"];
            let types = vec!["primitive_type", "type_identifier"];
            let variables = vec!["identifier"];
            let operators = vec!["+", "-", "+=", "++", "--", "-=", "*", "/", "*=", "/=", "@", "^", "!", "!=", "^=", "=", "==", "%", "%=", ","];
            let strings = vec!["string_literal", "char_literal"];
            let literals = vec!["number_literal"];
            let field_ids = vec!["field_identifier"];
            let comments = vec!["comment"];
            let preproc = vec!["preproc_def"];
            // let modules = vec!["namespace_identifier"];

            let pairs = [
                (keywords,"c_kw"),
                (types,"c_ty"),
                (variables,"c_vr"),
                (operators,"c_op"),
                (strings,"c_st"),
                (literals,"c_nl"),
                (field_ids,"c_fi"),
                (comments,"c_co"),
                // (modules,"c_mo"),
                (preproc,"c_pp"),
            ];
            for (toks,code) in pairs.iter() {
                for &kw in toks.iter() {
                    if let Some(id) = kind_map.get(kw) {
                        lang_classes.insert(*id, String::from(*code));
                    }
                }
            }
            classes_per_lang.push((vec![String::from("c")], lang_classes));
        }


        let sh = SyntaxHighlighter { shared_stylesheet, classes_per_lang };
        return sh;
    }

    fn get_lang_classes(&self, lang: &str) -> Option<&BTreeMap<u16,String>> {
        for (ks,v) in self.classes_per_lang.iter() {
            for k in ks.iter() {
                if k.to_lowercase() == lang.to_lowercase() {
                    return Some(v);
                }
            }
        }
        return None;
    }

    pub fn colorize(&self, code: &str, lang: &str) -> String {

        let src = code.as_bytes();

        let mut parser = Parser::new();

        let lang_lower = lang.to_lowercase();
        let ts_lang = match lang_lower.as_str() {
            "cpp" | "c++" => tree_sitter_cpp::language(),
            "c" => tree_sitter_c::language(),
            _ => panic!("unknown language {}", lang)
        };
        parser.set_language(ts_lang).expect(&format!("Error loading grammar for lang '{}'", lang));

        let parsed = parser.parse(code, None);

        let mut t = parsed.unwrap();
        let mut c = t.walk();
        let mut out = String::new();

        let mut ranges : BTreeMap<usize, (usize,u16)> = std::collections::BTreeMap::new();
        let mut visit_cb_insert_range = |node:&tree_sitter::Node| {
            ranges.insert(node.start_byte(), (node.end_byte(), node.kind_id()));
        };
        visit(&mut c,src, &mut visit_cb_insert_range);

        // Verify we have not messed up.
        // NOTE: We cannot have any overlapping ranges.
        let mut it1 = ranges.iter();
        let mut it2 = ranges.iter();
        it1.next();
        while let Some((b_start,(b_end,_))) = it1.next() {
            let (a_start,(a_end,_)) = it2.next().unwrap();
            // println!("a[{}:{}], b[{}:{}]", a_start,a_end, b_start,b_end);
            assert!(b_start >= a_end);
        }


        let lang_classes: &BTreeMap<u16,String> = self.get_lang_classes(&lang).expect(&format!("language '{}' not supported", lang));

        // Build final string.
        let mut out:String = String::new();
        let mut prev_end:usize = 0;
        for (start,(end,kind)) in ranges.iter() {

            if start - prev_end > 0 {
                let txt = std::str::from_utf8(&src[prev_end..*start]).unwrap();
                out = out + &txt;
            }

            if end - start > 0 {
                let txt = std::str::from_utf8(&src[*start..*end]).unwrap();
                out = out + &match lang_classes.get(kind) {
                    Some(class) => format!("<span class=\"{}\">{}</span>", &class, &txt),
                    None => txt.to_string() // wasteful, but allows match expr
                }
            }
            prev_end = *end;
        }

        if prev_end < src.len() {
                let txt = std::str::from_utf8(&src[prev_end..src.len()]).unwrap();
                out = out + &txt;
        }

        println!("final: {}", out);

        // Add proper whitespace at start of lines.
        let mut lines = out.split("\n");
        let mut lines1 = vec![];
        for line in lines {
            let mut space = 0;
            for c in line.chars() {
                if c == ' ' { space += 1; }
                else if c == '\t' { space += 4; }
                else { break }
            }

            let mut space_str = String::new();
            while space > 0 {
                if space >= 4 {
                    space_str += "&emsp;";
                    space -= 4;
                }
                else if space >= 2 {
                    space_str += "&ensp;";
                    space -= 2;
                }
                else if space >= 1 {
                    space_str += "&nbsp;";
                    space -= 1;
                }
            }
            lines1.push(space_str + line);
        }

        // Finally: stitch lines back together, inserting <br>'s
        let mut out = lines1.join("<br>\n");

        // out = out.replace("\n", "<br>\n");
        // let html = format!("<html><head><style>{}</style></head></body>\n{}\n</body></html>", self.shared_stylesheet, out);
        let html = format!("<div class=\"code\">{}</div>", out);

        return html;
    }

}


/*
fn download_treesitter_sources(dir: &str) -> Result<(), ureq::Error> {
    use std::io::copy;
    use std::io::Write;
    use std::fs::File;

    let files = vec!["parser.c", "scanner.c", "tree_sitter/parser.h"];
    for file in files {
        let body: String = ureq::get(&format!("https://raw.githubusercontent.com/tree-sitter/tree-sitter-javascript/master/src/{}", &file))
            .call().expect("failed http get")
            .into_string().expect("failed get into_string");
        let f: PathBuf = [dir, file].iter().collect();
        let mut fp = std::fs::OpenOptions::new()
                              .write(true)
                              .create(true)
                              .open(f.clone()).expect(&format!("failed to open file {}", f.display()));
        fp.write_all(body.as_bytes());
        // println!("body: {}", body);
    }
    Ok(())
}
*/
