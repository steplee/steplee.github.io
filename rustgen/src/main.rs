
use cc;
mod syntax_highlight;
mod markdown_parser;
mod index;

fn write_file(path: &str, data: &str) {
        use std::io::copy;
        use std::io::Write;
        use std::fs::File;
        let mut fp = std::fs::OpenOptions::new()
                                .write(true)
                                .create(true)
                                .truncate(true)
                                .open(path).expect(&format!("failed to open file {}", path));
        fp.write_all(data.as_bytes());
}

fn enumerate_posts(path: &str) -> Vec<String> {
    use std::fs;
    use std::path::Path;
    use std::ffi::OsStr;
    let mut out = vec![];
    for entry in fs::read_dir(path).expect(&format!("failed to read_dir '{}'", &path)) {

        if let Ok(entry) = entry {
            let fname_os = entry.file_name();
            let fname = fname_os.to_str().unwrap();
            if fname.ends_with(".md") || fname.ends_with(".MD") {
                // out.push(std::path::Path::new(path).join(entry.path()).to_str().unwrap().to_owned());
                out.push(entry.path().to_str().unwrap().to_owned());
            }
        }
    }

    println!("posts {:?}", out);

    return out;
}

static navbar_element: &'static str = r##"
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
        <a class="navbar-brand" href="#">Navbar</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarColor02" aria-controls="navbarColor02" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
        </button>

        <div class="collapse navbar-collapse" id="navbarColor02">
        <ul class="navbar-nav me-auto">
            <li class="nav-item">
            <a class="nav-link active" href="../index.html">Blog
                <span class="visually-hidden">(current)</span>
            </a>
            </li>
            <li class="nav-item">
            <a class="nav-link" href="https://github.com/steplee">GitHub</a>
            </li>

            <!-- This is not working -->
            <li class="nav-item dropdown">
            <a class="nav-link dropdown-toggle" data-bs-toggle="dropdown" href="#" role="button" aria-haspopup="true" aria-expanded="false" id="dropdownMenuButton1" data-toggle="dropdown1">Links</a>
            <div class="dropdown-menu" aria-labelledby="dropdownMenuButton1" id="dropdown1">
                <a class="dropdown-item" href="https://www.shadertoy.com/user/stephenl7797">ShaderToy</a>
                <a class="dropdown-item" href="https://www.shadertoy.com/user/stephenl7797">ShaderToy</a>
            </div>
            </li>

        </ul>
        </div>
    </div>
    </nav>
"##;

// Note: find_posts() does not populate any of the posts' `meta` member.
//       Instead that is done after parsing and creating the markdown for each post.
//       However we need the meta to create the index (it has the title and date),
//       so don't run this until after the markdown parsing step.
fn create_index_html(index: &index::BlogIndex, out_dir: &str) {
    for post in index.posts.iter() {

    }

}

fn main() {
    // println!("Hello, world!");


    // Testing syntax highlighting.
    /*
    let code = r#"
    #define x 1
std::string s = "hello";
    char c = '1';
static int w;
struct Foo{};
int wtice(int x) { if(1) {Foo x;} return x.a().y(2,2).z % 2.; } // com
        "#;

        let code_div = sh.colorize(&code, "cpp");

        let extra_part = r#"
<p>This is a paragraph.
It has a newline but no br.
Again.</p>

<p>This is a paragraph.<br>
It has a newline AND a br.<br>
Again.</p>

"#;
    */

    let sh = syntax_highlight::SyntaxHighlighter::new();

    let extra_head = r#"
    <link rel="stylesheet" href="./res/main.css">
    <link rel="stylesheet" href="./res/boostrap.darkly.css">
    <link rel="stylesheet" href="./res/code.css">
"#;

    write_file("out/res/code.css", &sh.shared_stylesheet);

    let out_path = "out";

    let mut index = index::find_posts("../posts", &out_path);

    for mut post in index.posts.iter_mut() {
        let html_path = &post.html_path;

        // Parse markdown.
        let mdd = markdown_parser::parse_markdown_document(&post.md_path, Some(&sh));

        // Fill in metadata on post object.
        post.meta.title = Some(mdd.meta.get("title").expect("meta must include title").clone());
        post.meta.publish_date = Some(mdd.meta.get("date").expect("meta must include title").clone());
        if let Some(tags) = mdd.meta.get("tags") {
            post.meta.tags = Some(tags.split(",").map(|s| s.to_string()).collect());
        }
        post.meta.modify_date = mdd.meta.get("modify_date").cloned();

        // Export html document.
        let mdd_html = mdd.html;
        let main_content = format!(r#"
<body>
    {navbar_element}

    <div id="mainImage"></div>

    <div class="container">
        {mdd_html}
    </div>

</body>
"#);
        let html = format!("<html><head>{extra_head}</head>{main_content}</html>");
        // let post_title = post.rsplit("/").next().unwrap();
        // let post_title = post_title.split(".").next().unwrap();
        // write_file(&format!("out/{}.html",post_title), &html);
        let html_path = &post.html_path;
        write_file(&html_path, &html);
    }


    // Write index document.
    let mut index_html = "<ul class=\"post-list\">\n".to_owned();
    for post in index.posts.iter_mut() {
        let post_title = &post.meta.title.as_ref().unwrap();
        let post_date = &post.meta.publish_date.as_ref().unwrap();
        index_html += &format!("<li class=\"post-item\"> <span class=\"post-date\">{post_date}</span> {post_title}</li>");
    }
    index_html += "</ul>\n";

    let index_content = format!(r#"
<body>
    {navbar_element}

    <div id="mainImage"></div>

    <div class="container">
        {index_html}
    </div>

</body>
"#);
    let html = format!("<html><head>{extra_head}</head>{index_content}</html>");
    write_file(&format!("{}/index.html", out_path), &html);

}
