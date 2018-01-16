import React from 'react';
import Link from 'gatsby-link';
import Helmet from 'react-helmet';
import CommonPage from '../templates/common';

import './project-list.css';

export default function ProjectList({ data }) {
  const { edges: projects } = data.allMarkdownRemark;
  const sections = [['wip','Work in Progress'], ['done','Finished']];
  return (
      <CommonPage>
        <section className="span12">
          {projects.filter(project => project.node.frontmatter.title.length > 0)
                .filter(project => project.node.frontmatter.status == 'wip')
                .map(({node: project}) => {
                  return (
                      <div className="project-block" key={project.id}>
                        <h1> <Link to={project.frontmatter.url}>
                          {project.frontmatter.title}
                        </Link> </h1>
                        <h2> {project.frontmatter.date} </h2>
                        <p>  {project.excerpt} </p>
                      </div>
                      );
                  })}
          </section>
        <section className="span12">
          {projects.filter(project => project.node.frontmatter.title.length > 0)
                .filter(project => project.node.frontmatter.status == 'done')
                .map(({node: project}) => {
                  return (
                      <div className="project-block" key={project.id}>
                        <h1> <Link to={project.frontmatter.url}>
                          {project.frontmatter.title}
                        </Link> </h1>
                        <h2> {project.frontmatter.date} </h2>
                        <p>  {project.excerpt} </p>
                      </div>
                      );
                  })}
          </section>
      </CommonPage>
      );
}

export const pageQuery = graphql`
  query ProjectQuery {
    allMarkdownRemark(
        sort: { order: DESC, fields: [frontmatter___date] },
        filter: {fileAbsolutePath: {regex: "/project-list/.*md$/"}}
    ) {
      edges {
        node {
          #excerpt(pruneLength: 250)
          id
          frontmatter {
            url
            title
            #date(formatString: "MMMM DD, YY")
            path
            status
          }
        }
      }
    }
  }`
