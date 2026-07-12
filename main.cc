#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "workflow_main.h"

namespace {

std::string workflow_config_path() {
    const char* home = std::getenv("HOME");
    if (home != NULL) {
        const std::string server_path = std::string(home) + "/files/workflow.conf";
        std::ifstream server_config(server_path.c_str());
        if (server_config.good()) {
            return server_path;
        }
    }
    return "files/workflow.conf";
}

std::vector<std::string> load_workflow_arguments() {
    const std::string path = workflow_config_path();
    std::ifstream input(path.c_str());
    if (!input) {
        std::cerr << "workflow config not found: " << path << '\n';
        std::cerr << "using the safe Flat/DEEP100K default\n";
        return {
            "--data-dir", "/anndata",
            "--method", "flat",
            "--queries", "2000",
            "--k", "10"
        };
    }

    std::vector<std::string> arguments;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line.erase(line.size() - 1);
        }
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        const size_t last = line.find_last_not_of(" \t");
        arguments.push_back(line.substr(first, last - first + 1));
    }

    if (arguments.empty()) {
        throw std::runtime_error("workflow config contains no arguments: " + path);
    }

    // qsub.sh copies the persistent result directory to $HOME/files. Make a
    // documented files/... output path independent of the PBS working dir.
    const char* home = std::getenv("HOME");
    if (home != NULL) {
        for (size_t i = 1; i < arguments.size(); ++i) {
            if (arguments[i - 1] == "--output" &&
                arguments[i].compare(0, 6, "files/") == 0) {
                arguments[i] = std::string(home) + "/" + arguments[i];
            }
        }
    }
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // The encrypted server test.sh invokes main without arguments. Local
        // ARM VM checks may still pass arguments directly to the executable.
        if (argc > 1) {
            return ann_workflow_main(argc, argv);
        }

        std::vector<std::string> storage;
        storage.push_back("ann_workflow");
        const std::vector<std::string> configured = load_workflow_arguments();
        storage.insert(storage.end(), configured.begin(), configured.end());

        std::vector<char*> argv;
        argv.reserve(storage.size());
        for (size_t i = 0; i < storage.size(); ++i) {
            argv.push_back(&storage[i][0]);
        }
        return ann_workflow_main(static_cast<int>(argv.size()), argv.data());
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
