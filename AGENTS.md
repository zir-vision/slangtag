# AGENTS.md file
 
## Dev environment tips
In your development container, you will not be able to run any GPU code (including tests). Stick to static analysis. If you need to run GPU code, you will have to ask the user to run it. You also do not have access to slangc. `cargo check` will work but it will make dummy shader compilations.