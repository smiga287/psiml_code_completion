
class ASTNode():
    def __init__(self, node_type, node_value, have_children: bool, have_sibling: bool):
        self.type = node_type
        self.value = node_value
        self.have_children = have_children
        self.have_sibling = have_sibling
    
    def __eq__(self, other):
        return self.type == other.type and self.value == other.value and self.have_children == other.have_children and self.have_sibling == other.have_sibling

    def __repr__(self):
        return f'[{self.type}, {self.value}, {self.have_children}, {self.have_sibling}]'

    def __str__(self):
        return f'[{self.type}, {self.value}, {self.have_children}, {self.have_sibling}]'