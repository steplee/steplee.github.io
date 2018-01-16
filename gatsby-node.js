/**
 * Implement Gatsby's Node APIs in this file.
 *
 * See: https://www.gatsbyjs.org/docs/node-apis/
 */


const path = require("path");

// Exported functions here will be run automatically by Gatsby

// uses Redux terminology -- "Bound Action Creators"  is a list of functions that gatsby provides
// graphql is a *callback function*
exports.createPages = ({ boundActionCreators, graphql }) => {
  const { createPage } = boundActionCreators;

  const blogPostTemplate = path.resolve(`src/templates/blog-post.js`);
  const noteTemplate = path.resolve(`src/templates/note.js`);

  /* -------------- BLOG ----------------- */
  
  var create_blog_posts = graphql(`{
    allMarkdownRemark(
        limit: 1000
        filter: {fileAbsolutePath: {regex: "/posts/.*md$/"}}
    ) {
      edges {
        node {
          excerpt(pruneLength: 250)
          fileAbsolutePath
          html
          id
          frontmatter {
            date
            path
            title
          }
        }
      }
    }
  }`).then(result => {
    if (result.errors) {
      console.log("Error:" + result.errors);
      return Promise.reject(result.errors);
    }


    // Actually create the pages
    result.data.allMarkdownRemark.edges
      .forEach(({node}) => {
        console.log("Making page " + node.frontmatter.path);
        createPage({
          path: node.frontmatter.path,
          component: blogPostTemplate,
          context: {}
        });
      });
  });

  /* -------------- NOTES ----------------- */

  /*
  var create_notes_pages = graphql(`{
    allMarkdownRemark(
        limit: 1000
        filter: {fileAbsolutePath: {regex: "/notes/.*md$/"}}
    ) { 
      edges {
        node {
          html
          id
          frontmatter {
            date
            path
            title
          }
        }
      }
    }
  }`).then(result => {
    if (result.errors) {
      console.log("Error for notes: " + result.errors);
      return Promise.reject(result.errors);
    }

    // Make the pages
    result.data.allMarkdownRemark.edges
      .forEach(({node}) => {
        console.log("Making note " + node.frontmatter.path);
        var path_prefix = "notes/";
        createPage({
          path: path_prefix + node.frontmatter.path,
          component: noteTemplate,
          context: {}
        });
      });
  });
  */



  return Promise.all([create_blog_posts]);
}
