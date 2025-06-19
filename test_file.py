def sum(a: int, b: int) -> int:
    if a < 10:
        if a > 13:
            return 5
        raise Exception("a must be >= 10")
    elif b > 12:
        return 9

def advanced_processing_logic(val_a: int, switch_b: bool, modifier_c: int) -> int:
    """
    A function with more complex logic to demonstrate test generation.
    It includes assignments within blocks, re-assignments, nested conditionals,
    and different types of return/raise statements.
    """
    temp_x = val_a * 2  # Initial assignment based on input

    if temp_x > 10:
        # Block 1: temp_x > 10
        temp_y = temp_x - 5  # Assignment within a conditional block

        if switch_b:
            # Block 1.1: temp_x > 10 AND switch_b is True
            if temp_y == 15:
                # Path where temp_x = 20 (val_a=10), switch_b=True -> temp_y=15
                return 101 # Expected: val_a=10, switch_b=True

            temp_x = temp_y + modifier_c  # Re-assignment of temp_x

            if temp_x < 25:
                # Path where original temp_x > 10, switch_b=True, temp_y != 15,
                # and new temp_x < 25
                return 202 # Example: val_a=11 (temp_x=22, temp_y=17), switch_b=True, modifier_c=5 (new temp_x=22)
            else:
                raise ValueError("Intermediate temp_x became too large") # Example: val_a=11, switch_b=True, modifier_c=10 (new temp_x=27)
        else:
            # Block 1.2: temp_x > 10 AND switch_b is False
            if modifier_c == 0:
                return 303 # Example: val_a=6 (temp_x=12), switch_b=False, modifier_c=0
            else:
                # This temp_y is from the outer scope of this 'else'
                return temp_y # Returns the value of temp_x - 5. Example: val_a=8 (temp_x=16, temp_y=11), switch_b=False, modifier_c=1
    else:
        # Block 2: temp_x <= 10
        if val_a == 5: # Original val_a
             # Path where temp_x = 10 (val_a=5)
            return 404 # Expected: val_a=5
        elif val_a < 0:
            raise TypeError("Negative val_a not allowed in this branch") # Expected: val_a = -1
        else:
            # Path where temp_x <= 10 but val_a is not 5 and not negative
            return None # Example: val_a=3 (temp_x=6)

def get_user_discount_percentage(age: int, is_member: bool) -> int:
    """
    Calculates a discount percentage based on age and membership status.
    - Ability: Handles integer inputs, boolean inputs, multiple conditions, direct returns.
    - Constraint: Assumes simple integer and boolean logic. Doesn't handle invalid input types
                  (e.g., negative age if not explicitly checked, though Z3 might find such a path).
    """
    if age < 18:
        return 5  # Child discount
    elif age >= 18 and age < 65:
        if is_member:
            return 20  # Adult member discount
        else:
            return 10  # Adult non-member discount
    elif age >= 65:
        if is_member:
            return 30  # Senior member discount
        else:
            return 25  # Senior non-member discount
    else:
        # This path should ideally be unreachable if age is always a valid number.
        # Your tool might show this as UNSAT or SAT depending on how `age` is constrained.
        return 0

def calculate_final_price(base_price: float, discount_percentage: float, tax_rate: float) -> float:
    """
    Calculates the final price of an item after applying a discount and then tax.
    """
    if not (0.0 <= discount_percentage <= 100.0):
        raise ValueError("Discount percentage must be between 0 and 100.")
    if not (tax_rate >= 0.0):
        raise ValueError("Tax rate cannot be negative.")

    discount_amount = base_price * (discount_percentage / 100.0)
    price_after_discount = base_price - discount_amount
    tax_amount = price_after_discount * tax_rate
    final_price = price_after_discount + tax_amount
    return final_price

def process_order_status(order_id: int, current_status: str) -> str:
    """
    Processes an order based on its current status.
    - Ability: Handles string inputs (though Z3 might treat them abstractly or require specific string theory),
               multiple string comparisons, different return values.
    - Constraint: String comparisons are exact. Z3's handling of strings might be basic
                  (e.g., equality/inequality) unless advanced string solving is enabled/supported.
                  The tool currently doesn't support string operations like `startswith` or `contains`
                  for path constraint generation.
    """
    if current_status == "NEW":
        # status_code = process_payment(order_id) # External call - Tool Limitation
        # if status_code == 200:
        #     return "PAYMENT_SUCCESSFUL"
        # else:
        #     return "PAYMENT_FAILED"
        # Simplified for current tool capabilities:
        if order_id > 0:
             return "PROCESSING"
        else:
             raise ValueError("Invalid Order ID for NEW order")

    elif current_status == "PROCESSING":
        return "SHIPPED"
    elif current_status == "SHIPPED":
        return "DELIVERED"
    elif current_status == "CANCELLED":
        return "REFUND_PENDING"
    else:
        raise TypeError(f"Unknown order status: {current_status}")


def configure_system_mode(cpu_threshold: int, memory_threshold: int, is_critical_service: bool) -> str:
    """
    Configures system mode based on resource thresholds and service criticality.
    - Ability: Demonstrates multiple integer inputs, boolean input, nested conditions,
               re-assignment of a variable (though not strictly necessary here, could be added),
               and returning different string literals.
    - Constraint: Logic is purely conditional. No loops or complex data structures involved.
                  The "mode" variable is assigned but not re-assigned in a way that tests SSA heavily.
    """
    mode = "NORMAL" # Initial assignment

    if is_critical_service:
        if cpu_threshold > 90 or memory_threshold > 85:
            mode = "CRITICAL_HIGH_LOAD"
            return mode # Path: Critical service, high load
        else:
            mode = "CRITICAL_NORMAL_LOAD"
            return mode # Path: Critical service, normal load
    else:
        # Non-critical service
        if cpu_threshold > 80 and memory_threshold > 70:
            # temp_var = cpu_threshold - 80 # Example of an intermediate assignment
            # if temp_var > 5:
            #    mode = "HIGH_PERFORMANCE"
            # else:
            #    mode = "PERFORMANCE"
            # Simplified for clarity:
            mode = "PERFORMANCE"
            return mode # Path: Non-critical, high load
        elif cpu_threshold < 20 and memory_threshold < 30:
            mode = "POWER_SAVER"
            return mode # Path: Non-critical, low load
        else:
            # Default for non-critical, moderate load
            return mode # Returns "NORMAL"

def check_file_access(file_size_kb: int, user_role: str) -> bool:
    """
    Checks if a user has access to a file based on its size and user role.
    - Ability: Handles integer and "string" (for role) inputs.
               Shows how `and` conditions are handled.
    - Constraint: `user_role` is treated as a simple value for comparison.
                  No actual file system interaction (tool limitation).
                  Limited number of roles for simplicity.
    """
    if user_role == "ADMIN":
        return True # Admins can access any file

    if user_role == "EDITOR":
        if file_size_kb < 10240: # Editors can access files < 10MB
            return True
        else:
            return False # File too large for editor
    
    if user_role == "VIEWER":
        if file_size_kb < 1024: # Viewers can access files < 1MB
            return True
        else:
            return False # File too large for viewer
            
    # Default deny for unknown roles or if no conditions met
    # (though with string comparisons, Z3 might need specific values for user_role to satisfy paths)
    raise PermissionError("Access denied or unknown role.")

def contradictory_conditions(a: int, b: bool) -> str:
    """
    A function with a clearly unsatisfiable path due to contradictory conditions.
    """
    if a > 100:
        if b:
            if a < 50:
                return "Path_A_Unreachable"
            return "Path_A_Reachable_B_True"
        else:
            return "Path_A_Reachable_B_False"
    else:
        if not b:
            return "Path_C_Not_B"
        else:
            return "Path_C_B"

def basic_float_ops(x: float, y: float) -> float:
    """
    Returns the result of (x + y) / (x - y).
    Raises ValueError if x == y to avoid division by zero.
    """
    if x == y:
        raise ValueError("x and y must not be equal")
    return (x + y) / (x - y)

