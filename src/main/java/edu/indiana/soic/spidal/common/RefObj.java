package edu.indiana.soic.spidal.common;

public class RefObj <T> {
    T value;

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }
}
